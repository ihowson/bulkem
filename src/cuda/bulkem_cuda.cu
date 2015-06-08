#include <pthread.h>
#include <stdio.h>
#include <sys/time.h>

#include "../bulkem.h"
#include "cxx11.h"


void *thread(void *void_args)
{
    fit_params *fp = (fit_params *)void_args;
    stream_main(fp);
    return 0;
}

void print_time_elapsed(struct timeval start_time, struct timeval end_time)
{
    double elapsed_time = (1000000.0 * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    printf("%f seconds elapsed\n", elapsed_time);
}

int bulkem_cuda(fit_params *fp)
{
    struct timeval start_time, end_time;
    int num_gpus;

    if (cudaSuccess != cudaGetDeviceCount(&num_gpus)) {
        printf("Failed to query CUDA devices2\n");
        return -4;
    }

    if (fp->verbose)
        printf("There are %d GPUs\n", num_gpus);

    chunk_reset();

    if (num_gpus != 1) {
        // error("We only support a single GPU right now");
        // return ScalarReal(-1000.0);

        // FIXME: return a more informative error message
        return -1;
    }

    // fire up some threads
    // FIXME: you might revise this in light of the multiple initialisations thing (you could just do one dataset at a time but multiple times; better cache locality, perhaps)
    if (fp->verbose)
        printf("Processing %d datasets simultaneously\n", NUM_THREADS);

    pthread_t threads[NUM_THREADS];

    gettimeofday(&start_time, 0);

    // launch threads
    for (unsigned i = 0; i < NUM_THREADS; i++)
    {
        int rc = pthread_create(&threads[i], NULL, thread, fp);

        if (rc)
        {
            // FIXME: do something smarter
            printf("THREAD LAUNCH FAILED rc=%d\n", rc);
            return 1;
        }
    }

    // join threads
    for (unsigned i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end_time, 0);
    printf("Processed %i chunks\n", fp->num_datasets);

    printf("cuda parallel: ");
    print_time_elapsed(start_time, end_time);

    return 0; // TODO: return some sort of success code
}


// FIXME: free everything that you malloced on both host and device - this is more important for R
