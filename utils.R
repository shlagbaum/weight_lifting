ranksTukeyHSD <- function(a, weights = NULL) {
    ra <- TukeyHSD(a);
    d <- ra[[1]];
    p = matrix(0, nrow = nrow(d), ncol = nrow(d));
    for (i in seq_len(nrow(p))) {
        k <- as.numeric(strsplit(rownames(d)[i], split = '-', fixed = TRUE)[[1]]);
        if (d[i, "diff"] >= 0) {
            p[k[1], k[2]] <- min(d[i, "p adj"], 1 - d[i, "p adj"]);
            p[k[2], k[1]] <- 1 - p[k[1], k[2]]
        } else {
            p[k[2], k[1]] <- min(d[i, "p adj"], 1 - d[i, "p adj"]);
            p[k[1], k[2]] <- 1 - p[k[2], k[1]]
        }
    }
    pvalue = unlist(summary(a))["Pr(>F)1"]*2;
    pU = 1 - pvalue;
    if (is.null(weights)) weights = rep(1.0, ncol(p));
    P = sapply(seq_len(ncol(p)), FUN = function(j) {
        W = weights;
        W[j] = 0;
        wtd.avg(p[,j], weights = W)
    })
    ans = rep(0, length(P));
    while (any(P > 0)) {
        i = which.max(P);
        ans[i] <- pU * P[i];
        P[i] <- 0;
        pU <- pU * (1 - ans[i])
    }
    list(TukeyHSD = ra, p = p, pvalue = pvalue, ranks = ans)
}
