#include "cuda_generate_values.cuh"

static __global__ void cuda_generate_then_add(size_t len, unsigned int seed,
                                             int *result)
{
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < len; i += stride)
  {
    // Keep the generated inputs local to this thread; only store the sum.
    const GeneratedValues values = cuda_generate_values(i, seed);
    result[i] = values.first + values.second;
  }
}
