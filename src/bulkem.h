#ifndef BULKEM_H
#define BULKEM_H

// Do not include any R or CUDA headers in here; the R and CUDA stuff needs to be compiled separately.

// To save time, we put a hard limit on the number of components that we can
// fit. This limit can be removed if necessary.
#define MAX_COMPONENTS 8

// TODO: Ditto for max observations. This limit really ought to be lifted; 10k obs isn't very large
#define MAX_OBSERVATIONS 1000000

// This ought to work well for most modern machines (at least, as of 2015).
// TODO: make it a parameter to the R interface
#define NUM_THREADS 8

enum fit_result {
    FIT_NOT_ATTEMPTED = 0, // default; dataset has not been fit yet
    FIT_SUCCESS,
    FIT_FAILED
    // DID_NOT_CONVERGE,
    // OTHER
};

typedef struct _invgauss_params_t
{
    double mu;
    double lambda;
    double alpha; // mixing proportion
} invgauss_params_t;

typedef struct _dataset {
    // Before fitting, these fields are filled in
    double *data;
    int num_observations;

    // After fitting, these fields are filled in
    enum fit_result fr;
    double final_loglik;
    invgauss_params_t init_params[MAX_COMPONENTS]; // initial parameters used for the fit
    invgauss_params_t fit_params[MAX_COMPONENTS];
    int num_iterations; // number of iterations used to produce the fit
    // TODO: return posterior probabilities
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


#ifdef __cplusplus
extern "C" {
#endif
    int bulkem_cuda(fit_params *fp);
    void stream_main(fit_params *fp);
#ifdef __cplusplus
}
#endif


#endif
