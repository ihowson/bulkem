library(bulkem)
library(mixsmsn)

context("todo context")

# test_that("two-component invgauss mixture model is fit well", {
#     up to here
#     generate a bunch of datasets and fit them
#     generate a summary of how good the fits are

#     fits <- bulkem(datasets, num.components=2, max.iters=10, random.inits=10, use.gpu=TRUE, epsilon=0.000001, verbose=TRUE)


test_that("package can be loaded and all functions called", {
    data(bmi)
    x1 <- bmi$bmi
    x2 <- c(0.1, 0.2, 0.3, 0.4)

    datasets <- list(x1, x2)
    # TODO: ensure that no exception is thrown
    # TODO: ensure that GPU was used
    fits <- bulkem(datasets, num.components=2, max.iters=10, random.inits=10, use.gpu=TRUE, epsilon=0.000001, verbose=TRUE)

    print(fits)

    expect_equal(length(fits), length(datasets))
})
