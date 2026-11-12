// Deterministic test data, not a statistical random-number generator.
static __global__ void cuda_generate_arrays(size_t len, unsigned int seed,
                                            int *vec1, int *vec2)
{
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < len; i += stride)
  {
    // Reduce before multiplying so even very large indices cannot overflow.
    const unsigned int x = static_cast<unsigned int>(i % 10000);
    const unsigned int first = ((seed % 10000) * x) % 10000;
    vec1[i] = static_cast<int>(first);
    vec2[i] = static_cast<int>((first * x) % 10000);
  }
}