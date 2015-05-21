#ifndef BULKEM_H
#define BULKEM_H

// Do not include any R or CUDA headers in here; the R and CUDA stuff needs to be compiled separately.

// To save time, we put a hard limit on the number of components that we can
// fit. This limit can be removed if necessary.
#define MAX_COMPONENTS 8

// TODO: Ditto for max observations. This limit really ought to be lifted; 10k obs isn't very large
#define MAX_OBSERVATIONS 10000

// This ought to work well for most modern machines (at least, as of 2015).
// TODO: make it a parameter to the R interface
#define NUM_THREADS 1

typedef struct _dataset {
    double *data;
    int num_observations;
} dataset;

typedef struct _fit_params {
    dataset *datasets;
    int num_datasets;
    int num_components;
    double epsilon;
    int verbose;
    int random_inits;
    int max_iters;
} fit_params;

typedef struct _invgauss_params_t
{
    double mu;
    double lambda;
    double alpha; // mixing proportion
} invgauss_params_t;
/*
typedef struct _fit_result {
    // success, error (too many iterations, nonconverging)

    int success;

    int num_iterations;
    initial parameters
    final parameters
    posterior probabilities (?)
    */


#ifdef __cplusplus
extern "C" {
#endif
    int bulkem_cuda(fit_params *fp);
    void stream_main(fit_params *fp);
#ifdef __cplusplus
}
#endif


#endif
