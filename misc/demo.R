#!/usr/bin/env Rscript

library(bulkem)
library(statmod)

USEGPU <- c(TRUE, FALSE)

D <- 100  # number of datasets
N <- 2000  # number of observations in each
NUM_REPEATS <- 10  # perform 10 random inits of each dataset

run <- function(xlist, use.gpu) {
    fits <- bulkem(datasets=xlist, use.gpu=use.gpu)

    f <- fits[[1]]

    print(sprintf("The first fit looks like mu=[%f %f], lambda=[%f %f], alpha=[%f %f]", f$mu[[1]], f$mu[[2]], f$lambda[[1]], f$lambda[[2]], f$alpha[[1]], f$alpha[[2]]))
}

make.dataset <- function(x) {
    alpha <- c(0.3, 0.7)  # mixture weights

    components <- sample(1:2, prob=alpha, size=N, replace=TRUE)
    mus <- c(1, 3)
    lambdas <- c(30, 0.2)

    x <- rinvgauss(n=N, mean=mus[components], shape=lambdas[components])
    return(x)
}

# make a list of 100 datasets
xlist <- lapply(1:D, make.dataset)

for (use.gpu in USEGPU) {
    print(sprintf('--- Using GPU: %d. %d datasets with %d observations in each, %d attempts per dataset.', use.gpu, D, N, NUM_REPEATS))
    time <- system.time(run(xlist, use.gpu=use.gpu))[3]
    print(sprintf("    Completed in %f seconds", time))
}
