#include <RcppArmadillo.h>
#include <RcppThread.h>
#include "IAggregator.hpp"
#include <boost/core/null_deleter.hpp>
#ifdef QUANTLIB_ENABLED
#include <ql/time/calendars/unitedstates.hpp>
#endif
#include <numeric>
#include <queue>
#include <cmath>
#include <fgt.hpp>

// [[Rcpp::export]]
RcppExport SEXP find_highest_cov_proj_cpp2(SEXP rx, SEXP rstdevs, SEXP ridx, SEXP weights) {
    try {
        NumericMatrix x(rx);
        NumericVector stdevs(rstdevs);
        IntegerVector idx(ridx);
        NumericVector w(weights);
        const double h = std::sqrt(2.0);
        int N = x.nrow(), M = x.ncol(), NN = 0;
        for (int i = 0; i < N; ++i) if (idx[i] == 0) ++NN;
        fgt::Matrix xx(NN, M);
        fgt::Vector ww(NN);
        for(int i = 0, ii = 0; i < N; ++i) {
            if (idx[i] == 0) {
                ww[ii] = w[i];
                for(int j = 0; j < M; ++j)
                    xx(ii,j) = x(i,j) / (stdevs[j] * h);
                ++ii;
            }
        }
        fgt::Vector covs = fgt::direct(xx, xx, 1.0, ww);
        double lambda = 0.0;
        for (int i = 0; i < NN; ++i)
            lambda += ww[i] * covs[i];
        lambda = std::sqrt(lambda);
        for (int i = 0; i < NN; ++i)
            covs[i] /= lambda;
        return Rcpp::wrap(List::create(Named("covs") = NumericVector(covs.begin(), covs.end()), Named("lambda") = lambda));
    } catch (std::exception &ex) {
        forward_exception_to_r(ex);
    } catch(...) {
        ::Rf_error("unkown c++ exception");
    }
    return R_NilValue;
}

RcppExport SEXP find_highest_cov_rows_cpp2(SEXP rx, SEXP rstdevs, SEXP ridx, SEXP rk, SEXP rweights) {
    try {
        NumericMatrix x(rx);
        NumericVector stdevs(rstdevs);
        NumericVector w(rweights);
        IntegerVector idx(ridx);
        int k = Rcpp::as<int>(rk);
        std::unordered_map<int, size_t> cluster_sizes;
        int N = x.nrow(), M = x.ncol();
        const double h = std::sqrt(2.0);
        
        for(int i = 0; i < N; ++i) 
            cluster_sizes[idx[i]]++;
        
        size_t L = cluster_sizes[k];
        fgt::Matrix y(L, M);
        fgt::Vector wy(L);
        for(int i = 0, ii = 0; i < N; ++i) {
            if (idx[i] == k) {
                wy[ii] = w[i];
                for(int j = 0; j < M; ++j)
                    y(ii,j) = x(i,j) / (stdevs[j] * h);
                ++ii;
            }
        }

        std::vector<double> covs(L, 0.0);
        std::vector<unsigned int> kdx(L);
        
        for(auto it = cluster_sizes.cbegin(); it != cluster_sizes.cend(); ++it) {
            size_t Nx = it->second;
            int kk = it->first;
            if (kk == k) continue;
            fgt::Matrix xx(Nx, M);
            fgt::Vector wx(Nx);
            for(int i = 0, ii = 0; i < N; ++i) {
                if (idx[i] == kk) {
                    wx[ii] = w[i];
                    for(int j = 0; j < M; ++j)
                        xx(ii,j) = x(i,j) / (stdevs[j] * h);
                    ++ii;
                }
            }
            fgt::Vector zx = fgt::direct(xx, xx, 1.0, wx);
            double lambda = 0.0;
            for (size_t i = 0; i < Nx; ++i)
                lambda += wx[i] * zx[i];
            lambda = std::sqrt(lambda);
            for (size_t i = 0; i < Nx; ++i)  
                wx[i] /= lambda;
            fgt::Vector zy = fgt::direct(xx, y, 1.0, wx);
            for (size_t i = 0; i < L; ++i) {
                if (zy[i] > covs[i]) {
                    covs[i] = zy[i];
                    kdx[i] = kk;
                }
            }
        } 
            
        return Rcpp::wrap(List::create(Named("covs") = NumericVector(covs.begin(), covs.end()), Named("nearest_clusters") = IntegerVector(kdx.begin(), kdx.end())));
    } catch (std::exception &ex) {
        forward_exception_to_r(ex);
    } catch(...) {
        ::Rf_error("unkown c++ exception");
    }
    return R_NilValue;
}
