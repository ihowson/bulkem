#ifndef LP_SUM_H
#define LP_SUM_H

// always launch this many threads per block
// 64 seems to be the minimum
// As we increase this, phase 2 time will increase but phase 1 will decrease. Until proper recursive sum reduction is implemented, this will require some tweaking.
#define LP_SUM_BLOCK_SIZE 128

// FIXME: give this a better name
typedef struct _commands
{
    unsigned num_commands;
    double *input[3];
    double *output[3];
} commands;

#ifdef __cplusplus
extern "C" {
#endif
    __global__ void lp_sum_kernel(const double *g_input, double *g_sum, unsigned n);
    __global__ void lp_fused_sum_kernel(commands c, unsigned n);
#ifdef __cplusplus
}
#endif

#endif /* LP_SUM_H */
