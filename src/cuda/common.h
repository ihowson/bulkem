#ifndef COMMON_H
#define COMMON_H


/*
#define CHUNK_BYTES (CHUNK_ENTRIES * sizeof(double))
#define POSTERIOR_CHUNK_BYTES (CHUNK_ENTRIES * sizeof(posterior_t))
#define ALL_CHUNK_BYTES (DATASET_ENTRIES * sizeof(posterior_t))
#define DATASET_ENTRIES (CHUNK_ENTRIES * NUM_CHUNKS)
#define DATASET_BYTES (sizeof(double) * DATASET_ENTRIES)

#define ALL_POSTERIOR_BYTES (sizeof(posterior_t) * DATASET_ENTRIES)

#define CONTROL_BYTES (NUM_CHUNKS * sizeof(control_t))

// this is the MaxOccupancy suggested block size for GTX660
#define BLOCK_SIZE 1024

// posterior probability for each component
typedef struct _posterior_t
{
    double component[NUM_MIXTURE_COMPONENTS];
} posterior_t;
*/


/*
typedef struct _invgauss_control_t
{
    unsigned iterations;
    double log_likelihood;

    // fitted distribution parameters
    invgauss_params_t params[NUM_MIXTURE_COMPONENTS];
} invgauss_control_t;

typedef invgauss_control_t control_t;
*/


#endif /* COMMON_H */
