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
bulkem(datasets, num.components=2, max.iters=100, random.inits=1, use.gpu=TRUE, epsilon=0.000001, verbose=FALSE)
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

- cuda accel
- random init

CUDA acceleration is most useful if you have a large number of datasets, require many random initialisations, or both. The degree of performance improvement that you achieve depends on what hardware you are running, but on the author's hardware (Intel i5-4460 and Geforce GTX 660) enabling CUDA yields a 30x speedup.

# Random init

The EM algorithm is sensitive to the choice of initial parameter estimates. \code{bulkem} implement the XYZ algorithm from ZYX paper, which operates roughly as follows for each mixture component:

- Sample \eqn{p + 1} observations from the dataset, where \eqn{p} is the number of parameters in each mixture component (so, four observations for inverse Gaussian mixture components)
- Generate a maximum likelihood estimate using those observations
- Use the MLE as the initial parameter estimates
}

\value{A list of mixEM objects. Each object describes a model that was fit by the EM algorithm. Each object in the list corresponds to a dataset in the input `datasets` list (that is, the \eqn{i}th object in the returned list is the model parameters for \eqn{i}th object in the `datasets` list.

Each mixEM object contains the following properties:

	\item{lambda}{The \eqn{\lambda} (shape) values for the model parameter estimates, returned as a vector.}
	\item{mu}{The \eqn{\mu} (shape) values for the model parameter estimates. This is returned as a vector, where each element of the vector corresponds to the mu estimates for a component of the mixture distribution (e.g. the second element of the vector is the mu estimate for the second mixture component.}
	\item{alpha}{The \eqn{\alpha} (component weight) values for the model parameter estimates. This is returned as a vector, where each element of the vector corresponds to the alpha estimates for a component of the mixture distribution (e.g. the second element of the vector is the alpha estimate for the second mixture component.}
	\item{init_lambda}{The \eqn{\lambda} values used as initial parameter estimates}
	\item{init_mu}{The \eqn{\mu} values used as initial parameter estimates}
	\item{init_alpha}{The \eqn{\alpha} values used as initial parameter estimates}
	\item{loglik}{The final log-likelihood of the model fit}
	\item{num_iterations}{The number of iterations of the EM algorithm required before convergence was achieved}
	\item{fit_success}{\code{TRUE} if the model was fit successfully; \code{FALSE} otherwise. The algorithm might fail to fit if convergence was not achieved within the allowed number of iterations or if an error was detected during fitting.}

For all parameter estimates returned as vectors, the \eqn{i}th element of the vector corresponds to the parameter estimate for the \eqn{i}th mixture component (e.g. the second element of the lambda vector is the lambda estimate for the second mixture component).
}

% TODO example: # Generate some sample data
\examples{

# Fit the BMI dataset using inverse Gaussian mixture models
# We need to try a few different initial conditions to get a good fit
library(mixsmsn)
data('bmi')
x <- list(bmi$bmi)

fits <- bulkem(x, random.inits=10)

print(paste0('fit llik: ', fit$llik, 'alpha: ', fit$alpha, ', lambda=', fit$lambda, ', mu=', fit$mu))

# plot the density function over the histogram
# modified from http://www.statmethods.net/graphs/density.html
h <- hist(x, breaks=40, col="red", main="BMI")
xfit <- seq(min(x), max(x), length=40)
yfit <- fit$alpha[1] * dinvgauss(xfit, mean=fit$mu[1], shape=fit$lambda[1]) + fit$alpha[2] * dinvgauss(xfit, mean=fit$mu[2], shape=fit$lambda[2])
yfit <- yfit * diff(h$mids[1:2]) * length(x) 
lines(xfit, yfit, col="blue", lwd=2)

}

% \keyword{TODO}
