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

    dataset *dataset_desc = calloc(num_datasets, sizeof(dataset));

    // convert from R representation to something we can use in CUDA code
    for (int i = 0; i < num_datasets; i++) {
        SEXP elem = VECTOR_ELT(datasets_, i);
        // TODO or should it be REALSXP?
        if (!isVector(elem)) {
            error("each dataset must be a vector of numerics");
        }

        double *vec = REAL(elem);
        int n = LENGTH(elem);

        // printf("\twe have %d entries. first three are %lf %lf %lf\n", n, vec[0], vec[1], vec[2]);

        dataset_desc[i].data = vec;
        dataset_desc[i].num_observations = n;
        dataset_desc[i].fr = FIT_NOT_ATTEMPTED;
    }

    fp.datasets = dataset_desc;
    fp.num_datasets = num_datasets;

    // make the call
    // int result = bulkem_cuda(&fp);
    bulkem_cuda(&fp);

    // bulkem_cuda() will update the datasets array with the results of
    // fitting. Go through those results and build a list to return to R.
    
    SEXP fits = PROTECT(allocVector(VECSXP, num_datasets));

    for (int i = 0; i < num_datasets; i++)
    {
        // each dataset gets a list containing results
        dataset *ds = &fp.datasets[i];
        SEXP dsr = allocVector(VECSXP, 9); // dsr is 'data set results'

        // note that child objects of 'fits' are automatically protected, so
        // no need to explicitly protect them

        SEXP lambda = allocVector(REALSXP, fp.num_components);
        SEXP mu = allocVector(REALSXP, fp.num_components);
        SEXP alpha = allocVector(REALSXP, fp.num_components);
        SEXP init_lambda = allocVector(REALSXP, fp.num_components);
        SEXP init_mu = allocVector(REALSXP, fp.num_components);
        SEXP init_alpha = allocVector(REALSXP, fp.num_components);

        for (int m = 0; m < fp.num_components; m++)
        {
            REAL(lambda)[m] = ds->fit_params[m].lambda;
            REAL(mu)[m] = ds->fit_params[m].mu;
            REAL(alpha)[m] = ds->fit_params[m].alpha;
            REAL(init_lambda)[m] = ds->init_params[m].lambda;
            REAL(init_mu)[m] = ds->init_params[m].mu;
            REAL(init_alpha)[m] = ds->init_params[m].alpha;
        }

        int fit_success = 0;
        if (ds->fr == FIT_SUCCESS)
            fit_success = 1;

        // NOTE: we set the names of the elements on the R side. If you change
        // the following ordering, make sure you change the R code too.
        SET_VECTOR_ELT(dsr, 0, lambda);
        SET_VECTOR_ELT(dsr, 1, mu);
        SET_VECTOR_ELT(dsr, 2, alpha);
        SET_VECTOR_ELT(dsr, 3, init_lambda);
        SET_VECTOR_ELT(dsr, 4, init_mu);
        SET_VECTOR_ELT(dsr, 5, init_alpha);
        SET_VECTOR_ELT(dsr, 6, ScalarReal(ds->final_loglik));
        SET_VECTOR_ELT(dsr, 7, ScalarInteger(ds->num_iterations));
        SET_VECTOR_ELT(dsr, 8, ScalarLogical(fit_success));

        // FIXME: this doesn't quite match the R interface

        SET_VECTOR_ELT(fits, i, dsr);
    }

    UNPROTECT(1); // unprotect 'fits'
    return fits;
}

