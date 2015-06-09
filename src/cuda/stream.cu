#include <pthread.h>
#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

#include <unistd.h>

#include "../bulkem.h"
#include "cxx11.h"
// TODO: this is only used for checkCudaErrors - try to remove it
#include "helper_cuda.h"
#include "lp_sum.h"

// return inverse Gaussian maximum likelihood parameter estimate for supplied samples
invgauss_params_t invgauss_maximum_likelihood(double *x, int n)
{
    // from http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Maximum_likelihood
    
    invgauss_params_t ret;
    
    //mu <- mean(x)
    double total = 0.0;
    for (int i = 0; i < n; i++)
        total += x[i];
    ret.mu = total / n;
    
    // lambda <- 1 / (1 / length(x) * sum(1 / x - 1 / mu))
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += 1.0 / x[i] - 1.0 / ret.mu;
        
    ret.lambda = 1.0 / (1.0 / n * sum);

    return ret;
}

// Generate a set of random initialisation parameters for the supplied data
void generate_invgauss_initial_params(double *x, int n, int num_components, uniform_rng &rng, invgauss_params_t *result)
{
    for (int m = 0; m < num_components; m++)
    {
        double sample[3]; // 3 is the number of parameters in maximum likelihood, plus one
    
        // take a random sample of the data
        // uses sampling with replacement for simplicity - shouldn't affect algorithm performance noticeably
        for (int i = 0; i < 3; i++)
        {
            // sample[i] = x[rng.rand()];
            // FIXME: remove the prior declaration if this is the interface you're going with
            // sample[i] = x[rng.rand_from_range(0, n - 1)];
            sample[i] = x[rng.rand_from_range()];
        }
    
        result[m] = invgauss_maximum_likelihood(sample, 3);
        result[m].alpha = 1.0 / num_components;
    }
}

__host__ __device__ double dinvgauss(double x, double mu, double lambda)
{
    // TODO would be nice to assert that x > 0 and lambda = 0

    double x_minus_mu = x - mu;
    // using the definition from Wikipedia
    return sqrt(lambda / (2 * CUDART_PI * pow(x, 3.0))) * exp((-lambda * x_minus_mu * x_minus_mu) / (2 * mu * mu * x));
}

// http://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
__device__ int getGlobalIdx_1D_1D()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void member_prob_kernel(
    const double *g_chunk,
    const invgauss_params_t *g_params,
    const int num_observations,
    const int num_components,
    double *g_member_prob,
    double *g_x_times_member_prob,
    double *g_lambda_sum_arg,
    double *g_log_sum_prob)
{
    // g_chunk: 1xN input
    // g_params: 1xM input
    // g_member_prob: MxN output
    // TODO document the rest

    /*
        x.prob <- dinvgauss(x_expanded, mean=mu_expanded, shape=lambda_expanded)  # N x 2 matrix
        weighted.prob <- alpha_expanded * x.prob  # per-component weighted sum of probabilities (Nx2)
        sum.prob <- rowSums(weighted.prob)  # second line of the T function from wikipedia (Nx1)
        member.prob <- weighted.prob / sum.prob
        */

    int t = getGlobalIdx_1D_1D();

    if (t >= num_observations)
    {
        // The recommended grid/block sizes often launch more kernels than we
        // have data for. Don't do anything if this is an excess kernel.
        return;
    }

    // FIXME: update to have flexible number of components
    double weighted_prob[MAX_COMPONENTS];
    double sum_prob = 0.0f;
    double x = g_chunk[t];

    #pragma unroll
    for (int m = 0; m < num_components; m++)
    {
        weighted_prob[m] = g_params[m].alpha * dinvgauss(x, g_params[m].mu, g_params[m].lambda);
        sum_prob += weighted_prob[m];
    }

    // used for convergence check; this is log(rowSums(alpha_expanded * x.prob)))
    g_log_sum_prob[t] = log(sum_prob);

    for (int m = 0; m < num_components; m++)
    {
        unsigned index = m * num_observations + t;

        double mp = weighted_prob[m] / sum_prob;
        // Elements of sum.prob can go to 0 if the density is low enough.
        // This causes divide-by-zero in the member.prob calculation. Replace
        // any NaNs with 0.
        if (isnan(mp))
            mp = 0.0;
        g_member_prob[index] = mp;

        // we don't use this until the end, but it's convenient to calculate here
        // it's the argument to colSums in:
        // mu.new <- colSums(x * member.prob) / member.prob.sum  # should be 1x2 matrix
        g_x_times_member_prob[index] = mp * x;

        // also used right at the end; this is the argument to colSums in:
        // lambda.new <- member.prob.sum / colSums(((x_expanded - mu_expanded) ^ 2 * member.prob) / (mu_expanded ^ 2 * x_expanded))
        // i.e. ((x_expanded - mu_expanded) ^ 2 * member.prob) / (mu_expanded ^ 2 * x_expanded)
        double x_minus_mu = x - g_params[m].mu;
        g_lambda_sum_arg[index] = (x_minus_mu * x_minus_mu * mp) / (g_params[m].mu * g_params[m].mu * x);
    }
}

void dump_dev_array(const char *msg, double *dev_array, int offset = 0)
{
    printf("tid %p %s: ", pthread_self(), msg);
    for (int i = offset; i < offset + 4; i++)
    {
        double val;

        checkCudaErrors(cudaMemcpyAsync(&val, &dev_array[i], sizeof(double), cudaMemcpyDeviceToHost, 0));
        checkCudaErrors(cudaStreamSynchronize(0));
        printf("%lf ", val);
    }

    printf("\n");
}

void dump_dev_value(const char *msg, double *dev_ptr, cudaStream_t stream)
{
    double val;

    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaMemcpyAsync(&val, dev_ptr, sizeof(double), cudaMemcpyDeviceToHost, 0));
    checkCudaErrors(cudaStreamSynchronize(0));
    printf("tid %p %s: %lf \n", pthread_self(), msg, val);
}

/* So that we only have one memcpy back from the device, we combine the results into this struct and copy them in one operation */
typedef struct _igresults
{
    double xmp_sum;
    double lambda_sum;
    double member_prob_sum;
} igresults;

void stream_main(fit_params *fp)
{
    invgauss_params_t *params_new;
    checkCudaErrors(cudaMallocHost(&params_new, sizeof(invgauss_params_t) * fp->num_components));
    invgauss_params_t *start_params;
    checkCudaErrors(cudaMallocHost(&start_params, sizeof(invgauss_params_t) * fp->num_components));
    igresults *host_igresults;
    checkCudaErrors(cudaMallocHost(&host_igresults, sizeof(igresults)));
    double *host_loglik;
    checkCudaErrors(cudaMallocHost(&host_loglik, sizeof(double)));

    cudaStream_t stream;
    // cudaSetDevice(0); // TODO: adjust when we have multiple GPUs; probably assign a new group of threads to each GPU
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // persistent device memory
    // allocated out here to avoid cudaMalloc in main loop
    // FIXME: need to move this to inner loop - realloc if it is too small
    double *dev_chunk; // N elements
    checkCudaErrors(cudaMalloc(&dev_chunk, sizeof(double) * MAX_OBSERVATIONS));

    invgauss_params_t *dev_params;
    checkCudaErrors(cudaMalloc(&dev_params, sizeof(invgauss_params_t) * fp->num_components));

    // FIXME: this needs to move to inner loop too
    double *dev_member_prob;
    checkCudaErrors(cudaMalloc(&dev_member_prob, sizeof(double) * fp->num_components * MAX_OBSERVATIONS));

    // FIXME: this needs to move to inner loop too
    double *dev_x_times_member_prob;
    checkCudaErrors(cudaMalloc(&dev_x_times_member_prob, sizeof(double) * fp->num_components * MAX_OBSERVATIONS));

    igresults *dev_igresults;
    checkCudaErrors(cudaMalloc(&dev_igresults, sizeof(igresults)));

    // FIXME: this needs to move to inner loop too
    double *dev_lambda_sum_arg;
    checkCudaErrors(cudaMalloc(&dev_lambda_sum_arg, sizeof(double) * fp->num_components * MAX_OBSERVATIONS));

    // FIXME: this needs to move to inner loop too
    double *dev_log_sum_prob;
    checkCudaErrors(cudaMalloc(&dev_log_sum_prob, sizeof(double) * MAX_OBSERVATIONS));

    double *dev_loglik;
    checkCudaErrors(cudaMalloc(&dev_loglik, sizeof(double)));

    // Lots of computations use 2D matrices.
    // - We don't use BLAS as there are no mat-mults, only elementwise mult/add
    // - We store in standard vectors to simplify expression
    // - We store columns grouped together as this is the common access pattern
    // - We often use for loops as (for the target dataset) N ~= 2000 and this is
    //   already more parallel than we have capacity for. We also have many
    //   datasets to run and the infrastructure to run tasks in parallel, which
    //   covers up many sins.

    unsigned chunk_id = chunk_get();
    while (chunk_id < fp->num_datasets)
    {
        if (fp->verbose)
            printf("thread %p chunk %d\n", pthread_self(), chunk_id);

        dataset *ds = &fp->datasets[chunk_id];
        int N = ds->num_observations;
        uniform_rng rng(0, N - 1);

        // determine kernel launch parameters
        // http://stackoverflow.com/a/25010560/591483
        int blockSize;      // The launch configurator returned block size 
        int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
        int gridSize;       // The actual grid size needed, based on input size 

        checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (const void *)member_prob_kernel, 0, N));
        // Round up according to array size 
        gridSize = (N + blockSize - 1) / blockSize; 

        if (fp->verbose)
            printf("using blockSize=%d gridSize=%d\n", blockSize, gridSize);

        invgauss_params_t best_init_params[MAX_COMPONENTS];
        invgauss_params_t best_final_params[MAX_COMPONENTS];
        double best_loglik = -INFINITY;
        int best_iterations = 0;
        fit_result best_result = FIT_FAILED;
        
        // copy chunk to device
        int chunk_bytes = ds->num_observations * sizeof(double);
        checkCudaErrors(cudaMemcpyAsync(dev_chunk, ds->data, chunk_bytes, cudaMemcpyHostToDevice, stream));

        // Set this within the loop to try a new set of initialisation parameters
        int init_again = 0;

        // for each random init
        for (int init = 0; init < fp->random_inits; init++)
        {
            init_again = 0;
            // init device parameters
            // FIXME: you should save the initial params for later analysis if this turns out to be the best solution
            generate_invgauss_initial_params(ds->data, ds->num_observations, fp->num_components, rng, start_params);

            /*
            printf("starting params are:\n");
            for (int m = 0; m < fp->num_components; m++)
                printf("\tlambda=%lf mu=%lf alpha=%lf\n", start_params[m].lambda, start_params[m].mu, start_params[m].alpha);

            printf("\n");
            */

            memcpy(params_new, start_params, sizeof(invgauss_params_t) * fp->num_components);

            bool converged = false;
            bool failed = false;
            double old_loglik = -INFINITY;

            // run EM algorithm
            unsigned iteration = 0; // FIXME: nasty. Better to explicitly count.
            while (converged == false && failed == false && iteration < fp->max_iters)
            {
                iteration++;
                checkCudaErrors(cudaMemcpyAsync(dev_params, params_new, sizeof(invgauss_params_t) * fp->num_components, cudaMemcpyHostToDevice, stream));
                // checkCudaErrors(cudaStreamSynchronize(stream));

                //////// PROCESS CHUNK

                // calculate member.prob
                // x.prob <- dinvgauss(x_expanded, mean=mu_expanded, shape=lambda_expanded)  # N x 2 matrix
                // weighted.prob <- alpha_expanded * x.prob  # per-component weighted sum of probabilities (Nx2)
                // sum.prob <- rowSums(weighted.prob)  # second line of the T function from wikipedia (Nx1)
                // member.prob <- weighted.prob / sum.prob

                // There are a whole bunch of operations on the data and
                // parameters that can be calculated in one pass. We do them all
                // here to minimise kernel launch overhead. This does make the
                // program flow a bit confusing.
                member_prob_kernel<<<gridSize, blockSize, 0, stream>>>(dev_chunk, dev_params, N, fp->num_components, dev_member_prob, dev_x_times_member_prob, dev_lambda_sum_arg, dev_log_sum_prob);
                // The remaining operations are all summations over various
                // outputs from this kernel.

                // have we converged?
                // TODO(perf): we could probably save some time by only doing this check once every few iterations - it's slow relative to the rest of the iteration time
                // log.lik <- sum(log(rowSums(alpha_expanded * x.prob)))
                lp_sum_kernel<<<1, LP_SUM_BLOCK_SIZE, LP_SUM_BLOCK_SIZE * sizeof(double), stream>>>(dev_log_sum_prob, dev_loglik, N);

                // copy new log-likelihood back
                checkCudaErrors(cudaMemcpyAsync(host_loglik, dev_loglik, sizeof(double), cudaMemcpyDeviceToHost, stream));
                checkCudaErrors(cudaStreamSynchronize(stream)); // wait for copy to complete

                // printf("old ll = %lf, new ll = %lf\n", old_loglik, *host_loglik);

                if (old_loglik > *host_loglik)
                {
                    // we're going backwards
                    printf("FAILED TO CONVERGE. Giving up.\n");
                    // do something more useful here
                    failed = true;
                    break;
                }

                double diff = *host_loglik - old_loglik;
                if (diff < fp->epsilon) {
                    converged = true;
                    break;
                }

                // didn't converge, continue optimising

                for (int m = 0; m < fp->num_components; m++)
                {
                    // set up inputs to fused_kernel
                    // this is similar to the command buffer to a DMA engine
                    commands c;
                    c.input[0] = dev_member_prob + m * N;
                    c.output[0] = &dev_igresults->member_prob_sum;

                    c.input[1] = dev_x_times_member_prob + m * N;
                    c.output[1] = &dev_igresults->xmp_sum;;

                    c.input[2] = dev_lambda_sum_arg + m * N;
                    c.output[2] = &dev_igresults->lambda_sum;

                    c.num_commands = 3;

                    // TODO(perf): the components are independent, so we could sum them from one kernel launch
                    // BLECH: we pass this all in through arguments. It would be nicer to pass input/output pairs but this would incur another host-to-device memcpy
                    lp_fused_sum_kernel<<<1, LP_SUM_BLOCK_SIZE, LP_SUM_BLOCK_SIZE * sizeof(double), stream>>>(c, N);

                    // determine new parameters
                    checkCudaErrors(cudaMemcpyAsync(host_igresults, dev_igresults, sizeof(igresults), cudaMemcpyDeviceToHost, stream));
                    checkCudaErrors(cudaStreamSynchronize(stream)); // wait for copy to complete

                    // TODO(perf): you could move this out of the loop and run 6 sums from one kernel launch
                    // it would also help to fuse all of the igresults together to save another memcpy
                    params_new[m].alpha = host_igresults->member_prob_sum / N;
                    params_new[m].mu = host_igresults->xmp_sum / host_igresults->member_prob_sum;
                    params_new[m].lambda = host_igresults->member_prob_sum / host_igresults->lambda_sum;
                }

                // check for parameter sanity
                // this is especially a project for very small datasets (< 10 observations)
                for (int m = 0; m < fp->num_components; m++)
                {
                    invgauss_params_t *p = &params_new[m];
                    if (isnan(p->alpha) || isnan(p->mu) || isnan(p->lambda))
                    {
                        printf("bogus! trying again with different init params\n");
                        printf("m = %d alpha = %lf mu = %lf lambda = %lf\n", m, p->alpha, p->mu, p->lambda);
                        init_again = 1;
                    }
                }

                if (init_again)
                    break;
                
                /*
                printf("thread %p iter %d\n", pthread_self(), iteration);
                for (int m = 0; m < fp->num_components; m++)
                {
                    invgauss_params_t *p = &params_new[m];
                    printf("\tcomp %d alpha=%lf mu=%lf lambda=%lf\n", m, p->alpha, p->mu, p->lambda);
                }
                */
                
                old_loglik = *host_loglik;
            }

            if (init_again)
                continue;
            // Disabling this gains about .5 seconds on a 5 second run
            
            /*
            printf("thread %p fit chunk %d after %d iterations\n", pthread_self(), chunk_id, iteration);
            for (int m = 0; m < fp->num_components; m++)
            {
                invgauss_params_t *p = &params_new[m];
                printf("\tcomp %d alpha=%lf mu=%lf lambda=%lf\n", m, p->alpha, p->mu, p->lambda);
            }
            */
            

            // printf("    init %d loglik = %lf\n", init, best_loglik);
            // we've converged; is this solution better than the last?
            if (*host_loglik > best_loglik)
            {
                best_result = FIT_SUCCESS;
                best_loglik = *host_loglik;
                best_iterations = iteration; // TODO: there might be an off-by-one here
                memcpy(best_final_params, params_new, sizeof(invgauss_params_t) * fp->num_components);
                memcpy(best_init_params, start_params, sizeof(invgauss_params_t) * fp->num_components);
            }
        }

        // printf("DONE: loglik = %lf\n", best_loglik);

        // Save the final fit results
        // TODO you could do this within the loop to remove some code
        ds->fr = best_result;
        ds->final_loglik = best_loglik;
        memcpy(ds->fit_params, best_final_params, sizeof(invgauss_params_t) * fp->num_components);
        memcpy(ds->init_params, best_init_params, sizeof(invgauss_params_t) * fp->num_components);
        ds->num_iterations = best_iterations;

        // Do the next dataset
        chunk_id = chunk_get();
    }

    // tidy up
    cudaFree(dev_chunk);
    cudaFree(dev_params);
    cudaFree(dev_member_prob);
    cudaFree(dev_x_times_member_prob);
    cudaFree(dev_igresults);
    cudaFree(dev_lambda_sum_arg);
    cudaFree(dev_log_sum_prob);
    cudaFree(dev_loglik);
    cudaFree(params_new);
    cudaFree(start_params);
    cudaFree(host_igresults);
    cudaFree(host_loglik);
}
