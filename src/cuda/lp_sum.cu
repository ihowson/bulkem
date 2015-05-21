/*
 * Low-parallelism sum
 *
 * If you've got a relatively small array to sum, kernel launch time dominates.
 * This kernel performs a sum reduction across an array with a minimum of
 * kernel launches (one). This is useful for code using Streams, where you can
 * find other work for the GPU to do at the same time.
 * 
 * You *must* launch this with gridSize=1. It manages grids internally. Use
 * as large or as small a blockSize as you wish. Larger blockSizes will
 * complete the sum faster. A smaller blockSize is useful if you're
 * bottlenecking on kernel launch time and want to keep the GPU busy for
 * longer (while you do other work).
 *
 * The sum result is placed in thread 0's g_sum output.
 */

#include <assert.h>
#include <stdio.h>

#include "lp_sum.h"

__global__ void lp_sum_kernel(
    const double *g_input,
    double *g_sum,
    unsigned n) // number of valid items in g_input
{
    if (gridDim.x != 1 || blockDim.x != LP_SUM_BLOCK_SIZE)
    {
        printf("lp_sum: invalid launch configuration %d %d\n", gridDim.x, blockDim.x);
        return;
    }

    // we assume that this is 0
    // TODO: is this the same one 
    __shared__ double intermediate[LP_SUM_BLOCK_SIZE];


    // We use two passes. On the first pass, we stride through the input by
    // blockDim and sum all of those elements. On the second pass, we use a
    // standard sum reduction to sum across each block.

    // first pass: stride through input
    unsigned t = threadIdx.x; // offset relative to stride position
    intermediate[t] = 0.0f; // FIXME: this shouldn't be necessary
    for (unsigned i = 0; i < (n / LP_SUM_BLOCK_SIZE + 1); i++)
    {
        // i is which chunk of input we're processing

        unsigned index = i * LP_SUM_BLOCK_SIZE + t; // index in input array
        if (index < n)
        {
            intermediate[t] += g_input[i * LP_SUM_BLOCK_SIZE + t];
        }
    }

    __syncthreads(); // wait until all threads have completed

    // second pass: sum the intermediate results
    // FIXME: this is wildly inefficient; most threads are idle
    if (t == 0)
    {
        double sum = 0.0f;
        for (unsigned i = 0; i < LP_SUM_BLOCK_SIZE; i++)
        {
            sum += intermediate[i];
        }

        *g_sum = sum;
    }
}

// Perform a number of summations at the same time
__global__ void lp_fused_sum_kernel(
    commands c,
    unsigned n) // number of valid data items to sum
{
    if (gridDim.x != 1 || blockDim.x != LP_SUM_BLOCK_SIZE)
    {
        printf("lp_sum: invalid launch configuration\n");
        return;
    }

    unsigned t = threadIdx.x; // offset relative to stride position

    // TODO: is this the same one for multiple kernel launches?
    __shared__ double intermediate[LP_SUM_BLOCK_SIZE];

    // We use two passes. On the first pass, we stride through the input by
    // blockDim and sum all of those elements. On the second pass, we use a
    // standard sum reduction to sum across each block.

    // iterate over each summation
    for (unsigned m = 0; m < c.num_commands; m++)
    {
        double *g_input = c.input[m];

        // first pass: stride through input
        intermediate[t] = 0.0f; // necessary for fused as we need to reset
        for (unsigned i = 0; i < (n / LP_SUM_BLOCK_SIZE + 1); i++)
        {
            // i is which chunk of input we're processing

            unsigned index = i * LP_SUM_BLOCK_SIZE + t; // index in input array
            if (index < n)
            {
                intermediate[t] += g_input[i * LP_SUM_BLOCK_SIZE + t];
            }
        }

        __syncthreads(); // wait until all threads have completed

        // second pass: sum the intermediate results
        // FIXME: this is wildly inefficient; most threads are idle
        if (t == 0)
        {
            double sum = 0.0f;
            for (unsigned i = 0; i < LP_SUM_BLOCK_SIZE; i++)
            {
                sum += intermediate[i];
            }

            double *g_sum = c.output[m];
            *g_sum = sum;
        }

        __syncthreads(); // wait until all threads have completed
    }
}
