#' @useDynLib bulkem

# Fit a number of finite mixture models using the Expectation Maximisation algorithm

# Maximum likelihood estimate of model parameters for data x
invgaussMaximumLikelihood <- function(x) {
    # from http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Maximum_likelihood
    mu <- mean(x)
    lambda <- 1 / (1 / length(x) * sum(1 / x - 1 / mu))

    result <- list(mu=mu, lambda=lambda)

    return(result)
}

#' @export
bulkem <- function(datasets, num.components=2, max.iters=100, random.inits=1, use.gpu=TRUE, epsilon=0.000001, verbose=FALSE) {
    # TODO: GPU datapath

    # TODO perhaps an interface where you pass NA to use.gpu means 'do your best'; if you pass TRUE or FALSE, require that setting

    if (use.gpu == TRUE) {
        # TODO: check input argument types
        #' @useDynLib bulkem bulkem_gpu_
        fits <- .Call(bulkem_host, datasets, num.components, max.iters, random.inits, epsilon, verbose)

        # TODO: fits$gpu <- TRUE

        for (index in range(length(datasets))) {
            names(fits[[index]]) <- c('lambda', 'mu', 'alpha', 'init_lambda', 'init_mu', 'init_alpha', 'loglik', 'num_iterations', 'fit_success')
        }

        # TODO if GPU failed, fall back to CPU
        # use.gpu <- FALSE
    }

    if (use.gpu == FALSE) {
        if (verbose) {
            print("Using CPU datapath")
        }

        fits <- list()

        # for each dataset...
        # print(sprintf('len %d', length(datasets)))
        for (index in 1:length(datasets)) {
            if (verbose) {
                print(sprintf('index %d', index))
            }
            x <- datasets[[index]]

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
                # print(paste0('fit llik: ', fit$llik, ', alpha: ', fit$alpha, ', lambda=', fit$lambda, ', mu=', fit$mu))

                if (is.null(best.fit)) {
                    best.fit <- fit
                } else if (!is.null(fit) && (fit$llik > best.fit$llik)) {
                    # print(paste0('found better ', fit$llik))
                    best.fit <- fit
                }
            }

            # save results
            fits[[index]] <- best.fit
        }
    }

    return(fits)
}


.onUnload <- function (libpath) {
    library.dynam.unload("bulkem", libpath)
}
