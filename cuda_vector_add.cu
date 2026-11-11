#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
#include <cuda_runtime.h>

static constexpr unsigned int THREADS_PER_BLOCK = 512;

static void check_cuda(cudaError_t error, const char* call, int line)
{
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at line %d in %s: %s\n",
                 line, call, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)

static __global__ void cuda_vector_add(size_t len, const int* vec1,
                                      const int* vec2, int* result)
{
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < len; i += stride) {
    result[i] = vec1[i] + vec2[i];
  }
}

// Deterministic test data, not a statistical random-number generator.
static __global__ void cuda_generate_arrays(size_t len, unsigned int seed,
                                           int* vec1, int* vec2)
{
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < len; i += stride) {
    // Reduce before multiplying so even very large indices cannot overflow.
    const unsigned int x = static_cast<unsigned int>(i % 10000);
    const unsigned int first = ((seed % 10000) * x) % 10000;
    vec1[i] = static_cast<int>(first);
    vec2[i] = static_cast<int>((first * x) % 10000);
  }
}

static bool verify_result(const int* device_result, size_t len,
                          unsigned int seed)
{
  // Bound host memory use even when the device arrays are large.
  std::vector<int> chunk(std::min(len, size_t(65536)));
  for (size_t offset = 0; offset < len;) {
    const size_t count = std::min(chunk.size(), len - offset);
    CUDA_CHECK(cudaMemcpy(chunk.data(), device_result + offset,
                          count * sizeof(int), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < count; ++j) {
      // Independent CPU reference; these products fit in 64 bits.
      const std::uint64_t x = (offset + j) % 10000;
      const int expected = static_cast<int>((seed * x) % 10000
                                            + (seed * x * x) % 10000);
      if (chunk[j] != expected) {
        std::fprintf(stderr, "Mismatch at index %zu: got %d, expected %d\n",
                     offset + j, chunk[j], expected);
        return false;
      }
    }
    offset += count;
  }
  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::fprintf(stderr, "USAGE: %s positive_length_of_vectors\n", argv[0]);
    return EXIT_FAILURE;
  }

  errno = 0;
  char* end = nullptr;
  const unsigned long long requested = std::strtoull(argv[1], &end, 10);
  if (argv[1][0] < '0' || argv[1][0] > '9' || *end != '\0'
      || errno == ERANGE || requested == 0
      || requested > std::numeric_limits<size_t>::max() / (3 * sizeof(int))) {
    std::fprintf(stderr, "ERROR: length must be a positive integer whose "
                         "three arrays fit in the address space\n");
    return EXIT_FAILURE;
  }

  const size_t num = static_cast<size_t>(requested);
  const size_t bytes = num * sizeof(int);
  const unsigned int seed = 111111;
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (3 * bytes > free_bytes) {
    std::fprintf(stderr, "ERROR: three arrays need %zu bytes; "
                         "only %zu bytes of device memory are free\n",
                 3 * bytes, free_bytes);
    return EXIT_FAILURE;
  }

  // Cap the grid to the device limit; the kernels cover any remainder by stride.
  const size_t blocks_needed = 1 + (num - 1) / THREADS_PER_BLOCK;
  const unsigned int block_count = static_cast<unsigned int>(
      std::min(blocks_needed, size_t(properties.maxGridSize[0])));
  std::printf("CUDA Vector Add\nDevice: %s\nLength: %zu\nSeed: %u\n",
              properties.name, num, seed);

  int* d_vec1 = nullptr;
  int* d_vec2 = nullptr;
  int* d_result = nullptr;
  CUDA_CHECK(cudaMalloc(&d_vec1, bytes));
  CUDA_CHECK(cudaMalloc(&d_vec2, bytes));
  CUDA_CHECK(cudaMalloc(&d_result, bytes));

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float elapsed_ms = 0.0f;

  CUDA_CHECK(cudaEventRecord(start));
  cuda_generate_arrays<<<block_count, THREADS_PER_BLOCK>>>(num, seed,
                                                          d_vec1, d_vec2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  // Check asynchronous execution errors as well as launch errors.
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  std::printf("Generate: %.3f ms\n", elapsed_ms);

  CUDA_CHECK(cudaEventRecord(start));
  cuda_vector_add<<<block_count, THREADS_PER_BLOCK>>>(num, d_vec1, d_vec2,
                                                     d_result);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  std::printf("Add: %.3f ms\n", elapsed_ms);

  const bool valid = verify_result(d_result, num, seed);
  if (valid) {
    std::printf("Verification passed for all %zu elements.\n", num);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_vec1));
  CUDA_CHECK(cudaFree(d_vec2));
  CUDA_CHECK(cudaFree(d_result));
  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
