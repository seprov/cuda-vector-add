#include "cuda_generate_values.cuh"

// Deterministic test data, not a statistical random-number generator.
static __global__ void cuda_generate_arrays(size_t len, unsigned int seed,
                                            int *vec1, int *vec2)
{
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < len; i += stride)
  {
    const GeneratedValues values = cuda_generate_values(i, seed);
    vec1[i] = values.first;
    vec2[i] = values.second;
  }
}
