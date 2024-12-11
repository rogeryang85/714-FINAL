#ifndef MATMUL_KERNELS_H
#define MATMUL_KERNELS_H

#include <cuda_fp16.h>
#include <mma.h>

namespace needle {
namespace cuda {

__device__ __forceinline__ void loadSmemA(half *smem, half *A, int M, int K, int ko);
__device__ __forceinline__ void loadSmemB(half *smem, half *B, int N, int K, int ko);

__global__ void MatmulKernel(const scalar_t *a, scalar_t *b, scalar_t *out, size_t M, size_t N, size_t K, size_t version);
__global__ void MatmulEff(half *A, half *B, half *C, int M, int N, int K);

} 
} 

#endif // MATMUL_KERNELS_H