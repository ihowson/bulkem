// R interface
// CUDA code is compiled separately. This file must not have any CUDA dependencies. 

#include <R.h>
#include <Rinternals.h>

#include "bulkem.h"

// FIXME: checkCudaErrors will exit() if an error is detected. This isn't acceptable when running under R; we should detect the error and throw an exception (??) instead

// In long-running loops, regularly run Rcpp::checkUserInterrupt(). This aborts your C++ if the user has pressed Ctrl + C or Escape in R.

// TODO Avoid calls to assert(), abort() and exit(): these will kill the R process, not just your C code. Instead, use error() which is equivalent to calling stop() in R.

//' @useDynLib bulkem bulkem_host
SEXP bulkem_host(SEXP datasets_, SEXP num_components_, SEXP max_iters_, SEXP random_inits_, SEXP epsilon_, SEXP verbose_) {
    //printf("Hello, world!\n");

    // printf("bulkem_cuda returns %d\n", bulkem_cuda());

    fit_params fp;
    fp.num_components = asInteger(num_components_);
    fp.max_iters = asInteger(max_iters_);
    fp.random_inits = asInteger(random_inits_);
    fp.epsilon = asReal(epsilon_);
    fp.verbose = asInteger(verbose_);

    // TODO better input validation of args other than 'datasets'

    // parse datasets
    if (!isNewList(datasets_)) {
        printf("FOO %d", __LINE__);
        error("'datasets' must be a list");
        // return NULL;
    }

    int num_datasets = LENGTH(datasets_);

    dataset *dataset_desc = malloc(sizeof(dataset) * num_datasets);

    // convert from R representation to something we can use in CUDA code
    for (int i = 0; i < num_datasets; i++) {
        SEXP elem = VECTOR_ELT(datasets_, i);
        // TODO or should it be REALSXP?
        if (!isVector(elem)) {
            error("each dataset must be a vector of numerics");
        }

        double *vec = REAL(elem);
        int n = LENGTH(elem);

        printf("\twe have %d entries. first three are %lf %lf %lf\n", n, vec[0], vec[1], vec[2]);

        dataset_desc[i].data = vec;
        dataset_desc[i].num_observations = n;
    }

    fp.datasets = dataset_desc;
    fp.num_datasets = num_datasets;

    // make the call
    // int result = bulkem_cuda(&fp);
    bulkem_cuda(&fp);

    // TODO return a new list
    
    // SEXP fits = PROTECT(allocVector(VECSXP, length of input datasets));

    // printf("make fits\n");
    
    SEXP fits = PROTECT(allocVector(VECSXP, 3));
    SET_VECTOR_ELT(fits, 0, ScalarInteger(1));
    SET_VECTOR_ELT(fits, 1, ScalarReal(238.1));
    SET_VECTOR_ELT(fits, 2, ScalarReal(123.4));

    UNPROTECT(1);
    return fits;
}

