\name{bulkem}
\alias{bulkem}
\title{Fit Finite Mixture Models to Many Datasets Using Expectation Maximisation}

\description{
	Fit finite mixture models composed of the inverse Gaussian distribution to
	a number of datasets. A random initialisation strategy helps with 
	difficult-to-fit datasets. CUDA acceleration can be used to reduce fit
	time dramatically.
}

\usage{
bulkem(datasets, num.components=2, max.iters=100, random.inits=1, use.gpu=TRUE,
	epsilon=0.000001, verbose=FALSE)
}

\arguments{
	\item{datasets}{A list of datasets to fit. Each element of the list must be a
	         vector of numbers.}
	\item{num.components}{The number of components in the fitted mixture models}
	\item{max.iters}{Maximum number of iterations of the EM algorithm}
	\item{random.inits}{Number of times to try to fit each dataset}
	\item{use.gpu}{\code{TRUE} to use an available CUDA GPU; \code{FALSE} to use the CPU}
	\item{epsilon}{If the difference between log-likelihood of iterations of EM is below this number, stop iterating. Higher epsilon speeds up fits but gives less accurate parameter estimates; lower epsilon gives more accurate parameter estimates but takes longer to fit.}
	\item{verbose}{If \code{TRUE}, dump additional debugging information to the console}
}

\details{
	\code{bulkem} fits finite mixture models composed of inverse Gaussian distributed components. It has two major advantages over existing packages:

	\itemize{
		\item CUDA (GPU) acceleration
		\item Random initialisation of parameters
	}

	CUDA acceleration is most useful if you have a large number of datasets, require many random initialisations, or both. The degree of performance improvement that you achieve depends on what hardware you are running. On the author's hardware (Intel i5-4460 and GeForce GTX 660) enabling CUDA yields a 30x speedup for large numbers of small datasets and 36x for large datasets.
}

\section{Random initialisation}{
	The EM algorithm is sensitive to the choice of initial parameter estimates. \code{bulkem} tries many different sets of initialisation parameters in order to find a global optimum. The initialisation strategy proceeds as follows:

	\enumerate{
		\item Sample \eqn{p + 1} observations from the dataset, where \eqn{p} is the number of parameters in each mixture component (so, three observations for each inverse Gaussian mixture component)
		\item Generate a maximum likelihood estimate using those observations
		\item Use the MLE as the initial parameter estimates
	}
}

\value{
	A list of mixEM objects. Each object describes a model that was fit by the EM algorithm. Each object in the list corresponds to a dataset in the input \code{datasets} list (that is, the \eqn{i}th object in the returned list is the model parameters for \eqn{i}th object in the \code{datasets} list.

	Each mixEM object contains the following properties:

	\item{lambda}{Vector of \eqn{\lambda} (shape) values for the model parameter estimates}
	\item{mu}{Vector of \eqn{\mu} (shape) values for the model parameter estimates}
	\item{alpha}{Vector of \eqn{\alpha} (component weight) values for the model parameter estimates}
	\item{init_lambda}{Vector of \eqn{\lambda} values used as initial parameter estimates}
	\item{init_mu}{Vector of \eqn{\mu} values used as initial parameter estimates}
	\item{init_alpha}{Vector \eqn{\alpha} values used as initial parameter estimates}
	\item{loglik}{The final log-likelihood of the model fit}
	\item{num_iterations}{The number of iterations of the EM algorithm required before convergence was achieved}
	\item{fit_success}{\code{TRUE} if the model was fit successfully; \code{FALSE} otherwise. The algorithm might fail to fit if convergence was not achieved within the allowed number of iterations or if an error was detected during fitting.}

	For all parameter estimates returned as vectors, the \eqn{i}th element of the vector corresponds to the parameter estimate for the \eqn{i}th mixture component (e.g. the second element of the lambda vector is the lambda estimate for the second mixture component).
}

\examples{

# Fit the BMI dataset using inverse Gaussian mixture models
# We need to try a few different initial conditions to get a good fit
library(bulkem)
library(mixsmsn)
library(statmod)
data('bmi')
x <- bmi$bmi
xlist <- list(x)

fits <- bulkem(xlist, random.inits=10)
fit <- fits[[1]]

print(paste0('fit llik: ', fit$llik, 'alpha: ', fit$alpha, ', lambda=',
	fit$lambda, ', mu=', fit$mu))

# plot the density function over the histogram
# modified from http://www.statmethods.net/graphs/density.html
h <- hist(x, breaks=40, col="red", main="BMI")
xfit <- seq(min(x), max(x), length=40)
yfit <- fit$alpha[1] * dinvgauss(xfit, mean=fit$mu[1], shape=fit$lambda[1]) +
	fit$alpha[2] * dinvgauss(xfit, mean=fit$mu[2], shape=fit$lambda[2])
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col="blue", lwd=2)


# Fit a large number of small datasets and compare how long CPU and GPU take
library(bulkem)
library(statmod)

USEGPU <- c(TRUE, FALSE)

D <- 100  # number of datasets
N <- 2000  # number of observations in each
NUM_REPEATS <- 10  # perform 10 random inits of each dataset

run <- function(xlist, use.gpu) {
    fits <- bulkem(datasets=xlist, use.gpu=use.gpu)

    f <- fits[[1]]

    print(sprintf("The first fit looks like mu=[%f %f], lambda=[%f %f],
    	alpha=[%f %f]", f$mu[[1]], f$mu[[2]], f$lambda[[1]], f$lambda[[2]],
    	f$alpha[[1]], f$alpha[[2]]))
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
    print(sprintf('--- Using GPU: %d. %d datasets with %d observations in
    each, %d attempts per dataset.', use.gpu, D, N, NUM_REPEATS))
    time <- system.time(run(xlist, use.gpu=use.gpu))[3]
    print(sprintf("    Completed in %f seconds", time))
}

# On my computer, this produces the following output:

# [1] "--- Using GPU: 1. 100 datasets with 2000 observations in each, 10
#     attempts per dataset."
# Processed 100 chunks
# cuda parallel: 0.423166 seconds elapsed
# [1] "The last fit looks like mu=[2.783511 1.015576], lambda=[0.210135
#     28.845700], alpha=[0.722182 0.277818]"
# [1] "    Completed in 0.511000 seconds"
# [1] "--- Using GPU: 0. 100 datasets with 2000 observations in each, 10
#     attempts per dataset."
# [1] "The last fit looks like mu=[1.015576 2.783510], lambda=[28.845905
#     0.210135], alpha=[0.277818 0.722182]"
# [1] "    Completed in 10.214000 seconds"

}
