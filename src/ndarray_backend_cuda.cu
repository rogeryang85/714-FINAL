#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <iostream>
#include <sstream>

namespace needle
{
  namespace cuda
  {

#define BASE_THREAD_NUM 256

#define TILE 4
    typedef float scalar_t;
    const size_t ELEM_SIZE = sizeof(scalar_t);

    struct CudaArray
    {
      CudaArray(const size_t size)
      {
        cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
        if (err != cudaSuccess)
          throw std::runtime_error(cudaGetErrorString(err));
        this->size = size;
      }
      ~CudaArray() { cudaFree(ptr); }
      size_t ptr_as_int() { return (size_t)ptr; }

      scalar_t *ptr;
      size_t size;
    };

    struct CudaDims
    {
      dim3 block, grid;
    };

    CudaDims CudaOneDim(size_t size)
    {
      /**
       * Utility function to get cuda dimensions for 1D call
       */
      CudaDims dim;
      size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
      dim.block = dim3(BASE_THREAD_NUM, 1, 1);
      dim.grid = dim3(num_blocks, 1, 1);
      return dim;
    }

#define MAX_VEC_SIZE 8
    struct CudaVec
    {
      uint32_t size;
      int32_t data[MAX_VEC_SIZE];
    };

    CudaVec VecToCuda(const std::vector<int32_t> &x)
    {
      CudaVec shape;
      if (x.size() > MAX_VEC_SIZE)
        throw std::runtime_error("Exceeded CUDA supported max dimesions");
      shape.size = x.size();
      for (size_t i = 0; i < x.size(); i++)
      {
        shape.data[i] = x[i];
      }
      return shape;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Fill call
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void FillKernel(scalar_t *out, scalar_t val, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = val;
    }

    void Fill(CudaArray *out, scalar_t val)
    {
      CudaDims dim = CudaOneDim(out->size);
      FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Compact and setitem cals
    ////////////////////////////////////////////////////////////////////////////////

    // Untility function to convert contiguous index i to memory location from strides

    __global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                  CudaVec strides, size_t offset)
    {
      /**
       * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
       * non-compact input a, to the corresponding item (at location gid) in the compact array out.
       *
       * Args:
       *   a: CUDA pointer to a array
       *   out: CUDA point to out array
       *   size: size of out array
       *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
       *   strides: vector of strides of out array
       *   offset: offset of out array
       */
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

      /// BEGIN SOLUTION
      if (gid >= size)
      {
        return;
      }
      size_t index = offset;
      size_t remaining = gid;

      for (int i = shape.size - 1; i >= 0; --i)
      {
        index += (remaining % shape.data[i]) * strides.data[i];
        remaining /= shape.data[i];
      }

      out[gid] = a[index];
      /// END SOLUTION
    }

    void Compact(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                 std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
       * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
       * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
       * the functions after this, however, you'll need to define these kernels as you see fit to
       * execute the underlying function.
       *
       * Args:
       *   a: non-compact represntation of the array, given as input
       *   out: compact version of the array to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *a* array (not out, which has compact strides)
       *   offset: offset of the *a* array (not out, which has zero offset, being compact)
       */

      // Nothing needs to be added here
      CudaDims dim = CudaOneDim(out->size);
      CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                             VecToCuda(strides), offset);
    }

    __global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                       CudaVec strides, size_t offset)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

      /// BEGIN SOLUTION
      if (gid >= size)
      {
        return;
      }
      size_t index = offset;
      size_t remaining = gid;

      for (int i = shape.size - 1; i >= 0; --i)
      {
        index += (remaining % shape.data[i]) * strides.data[i];
        remaining /= shape.data[i];
      }

      out[index] = a[gid];
    }

    void EwiseSetitem(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                      std::vector<int32_t> strides, size_t offset)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                                  VecToCuda(strides), offset);
    }

    __global__ void ScalarSetitemKernel(scalar_t val, scalar_t *out, size_t size, CudaVec shape,
                                        CudaVec strides, size_t offset)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid >= size)
        return;

      size_t index = offset;
      size_t remaining = gid;

      for (int i = shape.size - 1; i >= 0; --i)
      {
        index += (remaining % shape.data[i]) * strides.data[i];
        remaining /= shape.data[i];
      }

      out[index] = val;
    }

    void ScalarSetitem(size_t size, scalar_t val, CudaArray *out, std::vector<int32_t> shape,
                       std::vector<int32_t> strides, size_t offset)
    {
      CudaDims dim = CudaOneDim(size);
      ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                                   VecToCuda(strides), offset);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar operations
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      // Calculate the global index of the thread.
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] + b[gid];
    }

    void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      /**
       * Add together two CUDA arrays.
       * Args:
       *   a: Input array 'a' to be added
       *   b: Input array 'b' to be added
       *   out: Output array to store the result of 'a + b'
       */
      CudaDims dim = CudaOneDim(out->size);

      // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
      EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      // Calculate the global index of the thread.
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] + val;
    }

    void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      /**
       * Add a scalar value to every element of a CUDA array.
       * Args:
       *   a: Input array 'a'
       *   val: Scalar value to be added
       *   out: Output array to store the result of 'a + val'
       */
      CudaDims dim = CudaOneDim(out->size);

      // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a',
      // and store the result in array 'out'.
      ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar multiplication
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseMulKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] * b[gid];
    }

    void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarMulKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] * val;
    }

    void ScalarMul(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }
    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar division
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseDivKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] / b[gid];
    }

    void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarDivKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] / val;
    }

    void ScalarDiv(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }
    ////////////////////////////////////////////////////////////////////////////////
    // Scalar power
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void ScalarPowerKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = pow(a[gid], val);
    }

    void ScalarPower(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar maximum
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseMaximumKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = max(a[gid], b[gid]);
    }

    void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarMaximumKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = max(a[gid], val);
    }

    void ScalarMaximum(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar equality
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseEqKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] == b[gid];
    }

    void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarEqKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] == val;
    }

    void ScalarEq(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar greater-than-or-equal
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseGeKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] >= b[gid];
    }

    void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarGeKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] >= val;
    }

    void ScalarGe(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise unary operations: log, exp, tanh
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseLogKernel(const scalar_t *a, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = log(a[gid]);
    }

    void EwiseLog(const CudaArray &a, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
    }

    __global__ void EwiseExpKernel(const scalar_t *a, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = exp(a[gid]);
    }

    void EwiseExp(const CudaArray &a, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
    }

    __global__ void EwiseTanhKernel(const scalar_t *a, scalar_t *out, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = tanh(a[gid]);
    }

    void EwiseTanh(const CudaArray &a, CudaArray *out)
    {
      CudaDims dim = CudaOneDim(out->size);
      EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
    }
    ///////////////////////////Application of Templates///////////////////////////

    /**
     * In the code the follows, use the above template to create analogous elementise
     * and and scalar operators for the following functions.  See the numpy backend for
     * examples of how they should work.
     *   - EwiseMul, ScalarMul
     *   - EwiseDiv, ScalarDiv
     *   - ScalarPower
     *   - EwiseMaximum, ScalarMaximum
     *   - EwiseEq, ScalarEq
     *   - EwiseGe, ScalarGe
     *   - EwiseLog
     *   - EwiseExp
     *   - EwiseTanh
     *
     * If you implement all these naively, there will be a lot of repeated code, so
     * you are welcome (but not required), to use macros or templates to define these
     * functions (however you want to do so, as long as the functions match the proper)
     * signatures above.
     */

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar operations
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    // Matmul Helper
    ////////////////////////////////////////////////////////////////////////////////

    const int MI = 128;
    const int NI = 128;
    const int KI = 32;
    const int MII = 64;
    const int NII = 64;
    const int KII = 16;
    const int wmmaM = 16;
    const int wmmaN = 16;
    // const int wmmaK = 16;

#define MIN(a, b) (a) < (b) ? (a) : (b)

    __device__ __forceinline__ void loadSmemA(half *smem, half *A, int M, int K,
                                              int ko)
    {
      // load 128 * 32
      int by = blockIdx.y;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int tz = threadIdx.z;
      int tid = tz * 64 + ty * 32 + tx;
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        int logic_row = i * 32 + tid / 4;
        int logic_col = tid % 4 * 8;
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        col = col ^ (((row & 3) << 3));
        void *ptr = (void *)(smem + row * 64 + col);
        uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                smem_ptr),
            "l"(&A[(by * 128 + logic_row) * K + (ko * KI + logic_col)]), "n"(16),
            "r"(16));

        // uint32_t smem_ptr;

        // asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        //     "%0, smem_ptr; }\n"
        //     : "=r"(smem_ptr)
        //     : "l"(ptr));

        // asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
        // ::"r"(smem_ptr),
        //              "l"(&A[(by * 128 + logic_row) * K + (ko * KI + logic_col)]),
        //              "n"(16));
      }
    }

    __device__ __forceinline__ void predLoadSmemA(half *smem, half *A, int M, int K,
                                                  int ko, bool pred_guard)
    {
      // load 128 * 32
      int by = blockIdx.y;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int tz = threadIdx.z;
      int tid = tz * 64 + ty * 32 + tx;
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        int logic_row = i * 32 + tid / 4;
        int logic_col = tid % 4 * 8;
        int row = i * 16 + tid / 8;
        int col = tid % 8 * 8;
        col = col ^ (((row & 3) << 3));
        void *ptr = (void *)(smem + row * 64 + col);
        uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

        asm volatile(
            "{\n"
            " .reg .pred p;\n"
            " setp.ne.b32 p, %0, 0;\n"
            " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
            "}\n" ::"r"((int)pred_guard),
            "r"(smem_ptr),
            "l"(&A[(by * 128 + logic_row) * K + (ko * KI + logic_col)]), "n"(16));
      }
    }

    __device__ __forceinline__ void loadSmemB(half *smem, half *B, int N, int K,
                                              int ko)
    {
      // load 128 * 32
      int bx = blockIdx.x;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int tz = threadIdx.z;
      int tid = tz * 64 + ty * 32 + tx;
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        int logic_row = i * 32 + tid / 4;
        int logic_col = tid % 4 * 8;
        int row = i * 16 + tid / 8;
        int col = tid / 4 % 2 * 32 + tid % 4 * 8;
        col = col ^ (((row & 3) << 3));
        void *ptr = (void *)(smem + row * 64 + col);
        uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
                smem_ptr),
            "l"(&B[(bx * 128 + logic_row) * K + (ko * KI + logic_col)]), "n"(16),
            "r"(16));
      }
    }

    __device__ __forceinline__ void predLoadSmemB(half *smem, half *B, int N, int K,
                                                  int ko, bool pred_guard)
    {
      // load 128 * 32
      int bx = blockIdx.x;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int tz = threadIdx.z;
      int tid = tz * 64 + ty * 32 + tx;
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        int logic_row = i * 32 + tid / 4;
        int logic_col = tid % 4 * 8;
        int row = i * 16 + tid / 8;
        int col = tid / 4 % 2 * 32 + tid % 4 * 8;
        col = col ^ (((row & 3) << 3));
        void *ptr = (void *)(smem + row * 64 + col);
        uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

        asm volatile(
            "{\n"
            " .reg .pred p;\n"
            " setp.ne.b32 p, %0, 0;\n"
            " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
            "}\n" ::"r"((int)pred_guard),
            "r"(smem_ptr),
            "l"(&B[(bx * 128 + logic_row) * K + (ko * KI + logic_col)]), "n"(16));
      }
    }

    union Float4
    {
      float4 f4;
      float2 f22[2];
    };

    __device__ __forceinline__ void storeSmemC(half *C, float *smem, int M, int N)
    {
      // load 128 * 128
      int bx = blockIdx.x;
      int by = blockIdx.y;
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int tz = threadIdx.z;
      int tid = tz * 64 + ty * 32 + tx;
      for (int i = 0; i < 128; ++i)
      {
        int row = i;
        int col = tid;
        int scol = col ^ ((row & 3) << 3);
        (C[(by * 128 + row) * N + bx * 128 + col]) = (half)smem[row * 128 + scol];
      }
    }

    __device__ __forceinline__ void loadFragA(unsigned int *frag, half *smem,
                                              int ki)
    {
      // frag: [j, k]: [2, 2]
      // load 64x16
      int tx = threadIdx.x;
      int tz = threadIdx.z;

      //   load 16x16 at a time
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        int row = tz * 64 + i * 16 + tx / 16 * 8 + tx % 8;
        int col = ki * KII + tx / 8 % 2 * 8;
        col = row % 2 * 32 + col;
        row = row / 2;
        col = col ^ (((row & 3) << 3));
        void *ptr = (void *)(smem + row * 64 + col);
        // uint32_t smem_ptr;
        // asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64"
        //     "%0, smem_ptr; }\n"
        //     : "=r"(smem_ptr)
        //     : "l"(ptr));
        uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(frag[i * 4 + 0]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]),
              "=r"(frag[i * 4 + 3])
            : "r"(smem_ptr));
      }
    }

    __device__ __forceinline__ void loadFragB(unsigned int *frag, half *smem,
                                              int ki)
    {
      // frag: [j, k]: []
      // load 64x16
      int tx = threadIdx.x;
      int ty = threadIdx.y;

// load 16x16 at a time
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
        int row = ty * 64 + i * 16 + tx / 16 * 8 + tx % 8;
        int col = ki * KII + tx / 8 % 2 * 8;
        col = row % 2 * 32 + col;
        row = row / 2;
        col = col ^ (((row & 3) << 3));
        void *ptr = (void *)(smem + row * 64 + col);
        // uint32_t smem_ptr;
        // asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        //     "%0, smem_ptr; }\n"
        //     : "=r"(smem_ptr)
        //     : "l"(ptr));
        uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(frag[i * 4 + 0]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]),
              "=r"(frag[i * 4 + 3])
            : "r"(smem_ptr));
      }
    }

    __device__ __forceinline__ void storeAccum(float *ptr, float *frag)
    {
      // frag [r, c, _]: [2, 2, 2]
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int tz = threadIdx.z;
      // smem view is [128x128]
#pragma unroll
      for (int i = 0; i < 4; ++i)
      {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
          for (int r = 0; r < 2; ++r)
          {
            for (int c = 0; c < 2; ++c)
            {
              int row = tz * 64 + i * 16 + r * 8 + tx / 4;
              int col = ty * 64 + j * 16 + c * 8 + tx % 4 * 2;
              int scol = col ^ ((row & 3) << 3);
              ptr[row * 128 + scol] = frag[i * 32 + j * 8 + r * 4 + c * 2 + 0];
              ptr[row * 128 + (scol + 1)] =
                  frag[i * 32 + j * 8 + r * 4 + c * 2 + 1];
            }
          }
        }
      }
    }

    __device__ __forceinline__ void mmaSync(unsigned int *fragA,
                                            unsigned int *fragB, float *accum)
    {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
          : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
            "r"(fragB[0]), "r"(fragB[1]), "f"(accum[0]), "f"(accum[1]),
            "f"(accum[4]), "f"(accum[5]));

      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
          : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
            "r"(fragB[2]), "r"(fragB[3]), "f"(accum[2]), "f"(accum[3]),
            "f"(accum[6]), "f"(accum[7]));
    }

    __global__ void MatmulEff(half *A, half *B, half *C, int M, int N, int K)
    {
      // A is row-major
      // B is col-major
      // 128 threads [x, y, z] = [32, 2, 2]
      // threadblock mma: 128x128x32
      // warp mma: 64x64x16
      extern __shared__ uint8_t shared_storage[];
      half *SA1 = reinterpret_cast<half *>(shared_storage);
      half *SA2 = SA1 + MI * KI;
      half *SA3 = SA2 + MI * KI;
      half *SA4 = SA3 + MI * KI;
      half *SB1 = SA4 + MI * KI;
      half *SB2 = SB1 + NI * KI;
      half *SB3 = SB2 + NI * KI;
      half *SB4 = SB3 + NI * KI;
      half *SA[] = {SA1, SA2, SA3, SA4};
      half *SB[] = {SB1, SB2, SB3, SB4};
      float *SC = reinterpret_cast<float *>(shared_storage);

      unsigned int FragA[2][4 * 4]; // [4, 4]
      unsigned int FragB[2][4 * 4]; // [4, 4]

      float Accum[4 * 4 * 8] = {0.0}; // [4, 4, 8]

      // FragIteratorA frag_iter_A(SA[0]);

      // prologue
      for (int i = 0; i < 3; ++i)
      {
        loadSmemA(SA[i], A, M, K, i);
        loadSmemB(SB[i], B, N, K, i);
        asm volatile("cp.async.commit_group;\n" ::);
      }

      asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
      __syncthreads();

      loadFragA(FragA[0], SA[(0) % 4], 0);
      loadFragB(FragB[0], SB[(0) % 4], 0);

      for (int ko = 0; ko < K / KI; ko += 1)
      {
        // 64x64x16 mma for each warp
        loadFragA(FragA[1], SA[(ko) % 4], 1);
        loadFragB(FragB[1], SB[(ko) % 4], 1);
#pragma unroll
        for (int mii = 0; mii < MII / wmmaM; mii += 1)
        {
#pragma unroll
          for (int nii = 0; nii < NII / wmmaN; nii += 1)
          {
            // 16x16x16 for each wmma
            int n = (mii & 1) ? NII / wmmaN - 1 - nii : nii;
            mmaSync(&FragA[0][mii * 4], &FragB[0][n * 4], &Accum[mii * 32 + n * 8]);
          }
        }
        bool pred_guard = ko + 3 < K / KI;
        predLoadSmemA(SA[(ko + 3) % 4], A, M, K, ko + 3, pred_guard);
        predLoadSmemB(SB[(ko + 3) % 4], B, N, K, ko + 3, pred_guard);
        asm volatile("cp.async.commit_group;\n" ::);

        asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
        __syncthreads();

        // 64x64x16 mma for each warp
        loadFragA(FragA[0], SA[(ko + 1) % 4], 0);
        loadFragB(FragB[0], SB[(ko + 1) % 4], 0);
#pragma unroll
        for (int mii = 0; mii < MII / wmmaM; mii += 1)
        {
#pragma unroll
          for (int nii = 0; nii < NII / wmmaN; nii += 1)
          {
            // 16x16x16 for each wmma
            int n = (mii & 1) ? NII / wmmaN - 1 - nii : nii;
            mmaSync(&FragA[1][mii * 4], &FragB[1][n * 4], &Accum[mii * 32 + n * 8]);
          }
        }
      }

      storeAccum(SC, Accum);
      __syncthreads();
      storeSmemC(C, SC, M, N);
    }

    __global__ void MatmulKernel(const scalar_t *a, scalar_t *b, scalar_t *out, size_t M, size_t N, size_t K, size_t version)
    {
      if (version == 1)
      {
        // 1D parallelism
        size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
        if (gid >= M)
          return;

        for (size_t i = 0; i < K; ++i)
        {
          scalar_t tmp_sum = 0.0;
          for (size_t j = 0; j < N; ++j)
          {
            tmp_sum += a[gid * N + j] * b[j + i * K];
          }
          out[i + gid * K] = tmp_sum;
        }
      }
      else if (version == 2)
      {
        // 2D parallelism

        size_t col = blockDim.x * blockIdx.x + threadIdx.x;
        size_t row = blockDim.y * blockIdx.y + threadIdx.y;
        if (row < M && col < K)
        {
          scalar_t result = 0.0;
          for (size_t i = 0; i < N; ++i)
          {
            result += a[row * N + i] * b[col + i * K];
          }
          out[row * K + col] = result;
        }
      }
    }

    void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
                uint32_t P)
    {
      /**
       * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
       * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
       * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
       * over (i,j) entries in the output array.  However, to really get the full benefit of this
       * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
       * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
       * the CPU backend, here you should implement a single function that works across all size
       * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
       * implementations, this function here will largely just set up the kernel call, and you should
       * implement the logic in a separate MatmulKernel() call.
       *
       *
       * Args:
       *   a: compact 2D array of size m x n
       *   b: comapct 2D array of size n x p
       *   out: compact 2D array of size m x p to write the output to
       *   M: rows of a / out
       *   N: columns of a / rows of b
       *   P: columns of b / out
       */

      /// BEGIN SOLUTION
      size_t version = 1;
      if (version == 1)
      {
        // 1D parallel
        CudaDims dim = CudaOneDim(M);
        MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P, version);
      }
      else if (version == 2)
      {
        // 2D parallel
        dim3 block(16, 16);
        dim3 grid((P + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P, version);
      }
      else
      {
        // Final project - ours
        dim3 block(M / MI, N / NI);
        dim3 grid(32, 2, 2);
        half *a_half = reinterpret_cast<half *>(a.ptr);
        half *b_half = reinterpret_cast<half *>(b.ptr);
        half *out_half = reinterpret_cast<half *>(out->ptr);
        MatmulEff<<<grid, block>>>(a_half, b_half, out_half, M, N, P);
      }

      /// END SOLUTION
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Max and sum reductions
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, size_t reduce_size, size_t maximum)
    {
      size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
      if (gid >= maximum)
        return;
      scalar_t tmp_max = a[gid * reduce_size];
      for (size_t i = 0; i < reduce_size; ++i)
      {
        tmp_max = tmp_max > a[gid * reduce_size + i] ? tmp_max : a[gid * reduce_size + i];
      }
      out[gid] = tmp_max;
    }

    void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
       * for simplicity you can perform each reduction in a single CUDA thread.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */
      /// BEGIN SOLUTION
      CudaDims dim = CudaOneDim(out->size);
      ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
      /// END SOLUTION
    }

    __global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t reduce_size, size_t maximum)
    {
      size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
      if (gid >= maximum)
        return;
      scalar_t tmp_max = a[gid * reduce_size];
      for (size_t i = 1; i < reduce_size; ++i)
      {
        tmp_max += a[gid * reduce_size + i];
      }
      out[gid] = tmp_max;
    }

    void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
       * can perform each reduction in a single CUDA thread.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */
      /// BEGIN SOLUTION
      CudaDims dim = CudaOneDim(out->size);
      ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
      /// END SOLUTION
    }

  } // namespace cuda
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m)
{
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset)
        {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer); });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out)
        {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
