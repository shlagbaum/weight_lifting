library(tidyverse)
source(paste0(finexp, "/numerai/Sugeno.R"))

if (!is.loaded('find_highest_cov_rows_cpp2')) {
    dyn.load('/usr/local/lib/libfgt.so');
    dyn.load(paste0(finexp, '/Aggregator/libAggregatorRcpp.so'));
}

X <- read_delim("~/fgt/test/data/X.txt", delim = ' ', col_names = FALSE);
stdevs <- sapply(X, sd);
flog.threshold(DEBUG)
M <- KMeans_py(x = as.matrix(X), weights = rep(1,nrow(X)), stdevs = stdevs, min_cluster_size = 100, max_cluster_size = Inf, min_cos = 0.9, p = 2, ndim = 0, split = "fgt")
