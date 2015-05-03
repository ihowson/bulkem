# Fit a number of finite mixture models using the Expectation Maximisation algorithm

# Maximum likelihood estimate of model parameters for data x
invgaussMaximumLikelihood <- function(x) {
    # from http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Maximum_likelihood
    mu <- mean(x)
    lambda <- 1 / (1 / length(x) * sum(1 / x - 1 / mu))

    result <- list(mu=mu, lambda=lambda)

    return(result)
}


bulkem <- function(datasets, num.components=2, max.iters=100, random.inits=1, use.gpu=TRUE, epsilon=0.000001, verbose=FALSE) {
    # TODO: GPU datapath

    fits <- list()

    if (use.gpu) {
        print("GPU datapath not implemented, falling back to CPU")
    }

    print("Using CPU datapath")

    # for each dataset...
    for (name in names(datasets)) {
        if (verbose) {
            print(name)
        }
        x <- datasets[[name]]

        # RANDOMISED INITIALISATION
        # For j attempts, sample a few items from the dataset. Calculate the
        # maximum likelihood parameters and use those as initial parameters for a
        # mixture component attempt.

        best.fit <- NULL

        # for j replicates
        for (j in 1:random.inits) {
            if (verbose) {
                print(paste('    attempt', j))
            }
            initials <- list()

            # come up with initial conditions
            for (m in 1:num.components) {
                # take a subset of the data
                # modified from http://stackoverflow.com/a/19861866/591483
                items <- sample(1:length(x), 3)
                randomsubset <- x[items]

                ml <- invgaussMaximumLikelihood(randomsubset)
                ml$alpha <- 0.5  # start with equal mixing proportions

                initials[[m]] <- ml
            }

            # perform the fit
            fit <- invgaussmixEM(x, initials=initials, num.components=num.components, max.iters=max.iters, epsilon=epsilon)
            # print(paste0('fit llik: ', fit$llik, 'alpha: ', fit$alpha, ', lambda=', fit$lambda, ', mu=', fit$mu))

            if (is.null(best.fit)) {
                best.fit <- fit
            } else if (fit$llik > best.fit$llik) {
                # print(paste0('found better ', fit$llik))
                best.fit <- fit
            }
        }

        # save results
        fits[[name]] <- best.fit
    }

    return(fits)
}
