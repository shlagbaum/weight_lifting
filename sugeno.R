fcmw = reticulate::import_from_path("fcmw", paste(finexp, "python/fcmw/", sep = '/'));

if (!is.loaded('rcpp_sugeno_measure_cum')) {
    dyn.load('/usr/local/lib/libfgt.so');
    dyn.load(paste0(finexp, '/Aggregator/libAggregatorRcpp.so'));
}


FCMDsplit = function(x, w = NULL, sugeno = FALSE) {
    r = MaxContrastSplitR(x = x, w = w, sugeno = sugeno);
    u = matrix(0, nrow = length(x), ncol = 2);
    u[x < r$value,1] <- 1;
    u[x >= r$value,2] <- 1;
    m = fcmw$FCMWD1(n_clusters=as.integer(2));
    S = apply(u, MARGIN = 2, sum, na.rm = TRUE);
    if (min(S) > 1)
        u = m$fit(K = as.matrix(x, ncol = 1), uinit = u)
    else
        u = m$fit(K = as.matrix(x, ncol = 1))
    u
}

dissolve_cluster = function(X, k, weights, stdevs, idx, p = 1, nthreads = 4) {
    if (p == 2) {
        U <- .Call('find_highest_cov_rows_cpp2', as.matrix(X), as.numeric(stdevs), as.integer(idx), as.integer(k), as.numeric(weights));
        idx[idx==k] <- U$nearest_clusters
    } else {
        U <- .Call('find_highest_cov_rows_cpp', as.matrix(X), as.list(stdevs), as.integer(which(idx==k)), as.integer(which(idx!=k)), as.numeric(p), as.integer(nthreads));
        idx[idx==k] <- idx[U$row_max]
    }
    idx
}

dissolve_clusters = function(X, weights, stdevs, idx, min_cluster_size, min_cos, p = 1, R = 1, ndim = 0, split = 'minentropy', nthreads = 4) {
    tbl <- base::table(idx);
    done <- (min(tbl) >= min_cluster_size) || (min_cluster_size > 0 && length(tbl) < 3) || (length(tbl) < 2);
    while (!done) {
        j = which.min(tbl);
        flog.info("Dissolving cluster %s: %d obs...", names(tbl)[j], tbl[[j]]);
        k = as.integer(names(tbl)[j]);
        if (sum(idx!=k) > 0) {
            idx <- dissolve_cluster(X, k, weights=weights, stdevs=stdevs, idx=idx, p = p, nthreads = nthreads);
            tbl <- base::table(idx);
            done <- (min(tbl) >= min_cluster_size) || (min_cluster_size > 0 && length(tbl) < 3) || (length(tbl) < 2);
        } else done = TRUE;
    }
    z <- calc_avg_cosines(tbl, X, idx, weights, stdevs, p, ndim);
    w <- data.frame(idx = idx, weights = weights) %>% group_by(idx) %>% summarise(weights = sum(weights, na.rm = TRUE)) %>% as.data.frame();
    ac_max <- wtd.avg(z, w$weights);
    flog.debug("Covariances: %s, Avg = %f", paste(sprintf("%.3f", z), collapse = ','), ac_max);
    done = all(z >= min_cos) || (length(tbl) < 2);
    idx_max <- idx;
    while (!done) {
        idx_bak <- idx;
        old_max <- ac_max;
        for (j in which(z < min_cos)) {
            flog.info("Trying to dissolve cluster %s, cos = %f, %d obs...", names(tbl)[j], z[j], tbl[[j]]);
            k = as.integer(names(tbl)[j]);
            if (sum(idx!=k) > 0) {
                kdx = (idx == k);
                Xk = X[kdx,,drop = FALSE];
                wk = weights[kdx];
                P <- do_cluster_partition(df = Xk, weights = wk, stats = stdevs, p = p, R = R, min_cluster_size = min_cluster_size, max_cluster_size = Inf, min_cos = 0.0, ndim = ndim, split = split, nthreads = nthreads);
                tt <- base::table(P$idx);
                zz <- calc_avg_cosines(tt, Xk, P$idx, wk, stdevs, p, ndim);
                for (l in seq_along(zz)) {
                    m = as.integer(names(tt)[l]); 
                    if (zz[l] < min_cos) {
                        flog.info("Dissolving cluster %d(%d), cos = %f, %d obs...", k, m, zz[l], tt[[l]]);
                        if (p == 2) {
                            kk = max(as.integer(names(tbl))) + l;
                            idx[kdx][P$idx==m] <- kk;
                            U <- .Call('find_highest_cov_rows_cpp2', as.matrix(X), as.numeric(stdevs), as.integer(idx), as.integer(kk), as.numeric(weights));
                            idx[idx==kk] <- U$nearest_clusters
                        } else {
                            U <- .Call('find_highest_cov_rows_cpp', as.matrix(X), as.list(stdevs), as.integer(which(kdx)[P$idx==m]), as.integer(which(!kdx)), as.numeric(p), as.integer(nthreads)); 
                            idx[kdx][P$idx==m] <- idx[U$row_max];
                        }
                    } else {
                        kk = max(as.integer(names(tbl))) + l;
                        flog.info("Creating new sub-cluster %d out of %d and %d: %d obs, cos = %f", kk, k, m, sum(P$idx==m), zz[l]);
                        idx[kdx][P$idx==m] <- kk;
                    }
                }
                tbl <- base::table(idx);
                zz <- calc_avg_cosines(tbl, X, idx, weights, stdevs, p, ndim);
                w <- data.frame(idx = idx, weights = weights) %>% group_by(idx) %>% summarise(weights = sum(weights, na.rm = TRUE)) %>% as.data.frame();
                ac <- wtd.avg(zz, w$weights);
                flog.debug("Covariances: %s, Avg = %f", paste(sprintf("%.3f", zz), collapse = ','), ac);
                if (ac > ac_max) {
                    ac_max = ac;
                    idx_max = idx;
                }
                idx <- idx_bak;
                tbl <- base::table(idx);
            }
        }
        idx <- idx_max;
        tbl <- base::table(idx);
        z <- calc_avg_cosines(tbl, X, idx, weights, stdevs, p, ndim);
        w <- data.frame(idx = idx, weights = weights) %>% group_by(idx) %>% summarise(weights = sum(weights, na.rm = TRUE)) %>% as.data.frame();
        ac_max <- wtd.avg(z, w$weights);
        flog.debug("Covariances: %s, Avg = %f", paste(sprintf("%.3f", z), collapse = ','), ac_max);
        done = (ac_max <= old_max) || all(z >= min_cos) || (length(tbl) < 2);
        if (ac_max <= old_max) {
            ac_max = old_max;
            idx <- idx_bak;
            flog.debug("Cannot dissolve further. Avg cov = %f", ac_max);
        } else {
            flog.debug("Dissolving further. Avg cov = %f, progress = %f", ac_max, ac_max - old_max);
        }
    }
    idx
}

do_cluster_partition = function(df, weights, stats, p = 1, R = 1, min_cluster_size = 500, max_cluster_size = 25000, min_cos = 0.5, ndim = 0, split = 'maxcontrast', nthreads = 4) {
    if (is.matrix(df)) {
        x = df;
        stdevs = stats;
        cols = colnames(df);
        w = na.fill(weights, 0);
    } else {
        cols = unlist(stats[['factors']]);
        stdevs = stats[['stdevs']];
        x = as.matrix(df[,cols]);
        w = na.fill(if (is.character(weights)) df[[weights]] else weights, 0);
    }
    idx = as.integer(rep(0, NROW(x)));
    ncluster = 0;
    done = FALSE;
    # w = w / sum(w);
    centers <- list();
    while (!done) {
        ncluster = ncluster + 1;
        jdx <- idx == 0;
        flog.info("Cluster %d: extracting from %d obs...", ncluster, sum(jdx));
        #out <- .Call('find_highest_cov_row_cpp', as.matrix(x), as.list(stdevs), as.integer(idx), as.numeric(p), as.integer(nthreads));
        #jdx[out$row_max] <- FALSE;
        S <- mean(w[jdx], na.rm = TRUE);
        if (!is.finite(S)) S = 0.0;
        ww = if (S < 1e-14) rep(1, length(w)) else w / S;
        if (grepl('fgt', tolower(split))) {
            j <- which(idx == 0);
            out <- .Call('find_highest_cov_proj_cpp2', as.matrix(x), as.numeric(stdevs), as.integer(idx), ww);
            if (ndim > 0) out$covs <- out$covs ^ (1/ndim);
            u <- FCMDsplit(out$covs, ww[j], sugeno = FALSE);
            k <- apply(u, MARGIN = 1, which.max);
            jj <- j[k == 2];
            if (length(jj) == 0) {
                jj <- j[k == 1];
            }
            if (length(jj) > max_cluster_size) {
                jj <- j[j != jj];
                if (length(jj) > max_cluster_size) {
                    jj = j[head(order(out$covs[j], decreasing = TRUE), max_cluster_size)];
                }
            }
            j = jj;
            flog.info("fgt: cluster size: %d", length(j));
        } else {
            out <- .Call('find_highest_cov_proj_cpp', as.matrix(x), as.list(stdevs), as.integer(idx), as.numeric(p), ww, as.integer(nthreads));
            if (ndim > 0) out$covs <- out$covs ^ (1/ndim);
        
            dst = sqrt(1 - out$covs[jdx]);
            ww = w[jdx];
        
            if (grepl('maxcontrast', tolower(split))) {
                if (grepl('med', tolower(split))) {
                    r = MaxContrastSplitRmed(dst, ww, sugeno = TRUE);
                } else {
                    r = MaxContrastSplitR(dst, ww, sugeno = TRUE);
                }
            } else
            if (split == 'minentropy') {
                r = MinEntropySplit(dst, ww);
            } else 
            if (split == 'maxentropy') {
                r = MinEntropySplitR(dst, ww);
                if (!(which.min(r$H) %in% c(1, sum(jdx)))) {
                    plot(r$H, type='l');
                    browser();
                }
            } else {
                if (split == 'sugeno') {
                    r = IntegrateSugeno(dst, ww, lambda = NULL, R = R);
                } else {
                    r = IntegrateChoquet(dst, ww, lambda = NULL, R = R);
                    r$at = 0;
                }
            }
            mid_cov = 1 - r$value^2;
            j = which(out$covs >= mid_cov);
            if (length(j) > max_cluster_size) {
                j = j[head(order(out$covs[j], decreasing = TRUE), max_cluster_size)];
            }
            flog.info("Value: %f, at = %d, lambda = %f, cluster size: %d", mid_cov, r$at, r$lambda, length(j));
        }
        idx[j] <- ncluster;
        centers[[ncluster]] <- out;
        done = (sum(idx==0) < min(100, min_cluster_size)) || (length(j) < 1);
        if (done) {
            #out <- .Call('find_highest_cov_row_cpp', as.matrix(x), as.list(stdevs), as.integer(idx), as.numeric(p), as.integer(nthreads));
            jdx = idx == 0;
            if (sum(jdx, na.rm = TRUE) > 0) {
                if (p == 2) 
                    out <- .Call('find_highest_cov_proj_cpp2', as.matrix(x), as.numeric(stdevs), as.integer(idx), w / mean(w[jdx], na.rm = TRUE))
                else 
                    out <- .Call('find_highest_cov_proj_cpp', as.matrix(x), as.list(stdevs), as.integer(idx), as.numeric(p), w / mean(w[jdx], na.rm = TRUE), as.integer(nthreads));
                if (ndim > 0) out$covs <- out$covs ^ (1/ndim);
                #if (out$row_max > 0) {
                    idx[jdx] <- ncluster + 1;
                    centers[[ncluster+1]] <- out;
                #}
            }
        }
    }
    idx <- dissolve_clusters(X = x, weights = w, stdevs = stdevs, idx = idx, min_cluster_size = min_cluster_size, min_cos = min_cos, p = p, R = R, ndim = ndim, split = split, nthreads = nthreads);
    centers <- lapply(base::split(x = as.data.frame(cbind(w, x)), f = idx), function(y) {
        x = as.matrix(y[, -1]);
        w = na.fill(as.numeric(y[,1]), 0);
        matrix(apply(x, MARGIN = 2, wtd.avg, weights = w), nrow = 1)
    }) %>% do.call(rbind, args = .);
    flog.info("Done do_cluster_partition: %d clusters, %s", NROW(centers), paste(as.character(as.numeric(table(idx))), collapse = ','));
    list(centers = as.matrix(centers), idx = idx)
}

