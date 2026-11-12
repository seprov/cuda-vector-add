static __global__ void cuda_vector_add(size_t len, const int *vec1,
                                       const int *vec2, int *result)
{
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < len; i += stride)
  {
    result[i] = vec1[i] + vec2[i];
  }
}
