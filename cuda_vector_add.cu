#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <cuda_runtime.h>
#include "cuda_check.cu"
#include "composites/fused.cuh"
#include "composites/galois.cuh"
#include "composites/split.cuh"

// Keep compiling just this file; include the method implementations here.
#include "husks/fused.cu"
#include "husks/galois.cu"
#include "husks/split.cu"
#include "stubble_burning/fused.cu"
#include "stubble_burning/split.cu"

static constexpr unsigned int THREADS_PER_BLOCK = 512;

// Originally built with --use_fast_math; these kernels only use integer math.
// Original size-limit note: 1_000_000_000 works, 10_000_000_000 does not work.
// In separate mode, the three device arrays require 12 GB and 120 GB, respectively.

int main(int argc, char *argv[])
{
  ////////////
  // set up
  ////////////

  // check command line
  if ((argc != 2 && argc != 3) ||
      (argc == 3 && std::strcmp(argv[2], "separate") != 0 &&
                    std::strcmp(argv[2], "fused") != 0 &&
                    std::strcmp(argv[2], "galois") != 0))
  {
    std::fprintf(stderr, "USAGE: %s positive_length_of_vectors [separate|fused|galois]\n",
                 argv[0]);
    return EXIT_FAILURE;
  }

  const bool fused = argc == 3 && std::strcmp(argv[2], "fused") == 0;
  const bool galois = argc == 3 && std::strcmp(argv[2], "galois") == 0;
  const size_t array_count = (fused || galois) ? 1 : 3;

  errno = 0;
  char *end = nullptr;
  const unsigned long long requested = std::strtoull(argv[1], &end, 10);
  if (argv[1][0] < '0' || argv[1][0] > '9' || *end != '\0' || errno == ERANGE || requested == 0 || requested > std::numeric_limits<size_t>::max() / (array_count * sizeof(int)))
  {
    std::fprintf(stderr, "ERROR: length must be a positive integer whose "
                         "device arrays fit in the address space\n");
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
  if (array_count * bytes > free_bytes)
  {
    std::fprintf(stderr, "ERROR: device arrays need %zu bytes; "
                         "only %zu bytes of device memory are free\n",
                 array_count * bytes, free_bytes);
    return EXIT_FAILURE;
  }

  // Cap the grid to the device limit; the kernels cover any remainder by stride.
  const size_t blocks_needed = 1 + (num - 1) / THREADS_PER_BLOCK;
  const unsigned int block_count = static_cast<unsigned int>(
      std::min(blocks_needed, size_t(properties.maxGridSize[0])));
  std::printf("CUDA Vector Add\nDevice: %s\nLength: %zu\nSeed: %u\n",
              properties.name, num, seed);
  std::printf("Mode: %s\n", fused ? "fused" : (galois ? "galois" : "separate"));

  std::unique_ptr<Composite> obj;
  if (fused)
  {
    obj = std::make_unique<FusedComposite>(num, seed, block_count,
                                           THREADS_PER_BLOCK);
  }
  else if (galois)
  {
    obj = std::make_unique<GaloisComposite>(num, seed, block_count,
                                            THREADS_PER_BLOCK);
  }
  else
  {
    obj = std::make_unique<SplitComposite>(num, seed, block_count,
                                           THREADS_PER_BLOCK);
  }
  obj->husk();

  // copy from GPU to CPU and verify
  const bool valid = obj->verify_result();
  if (valid)
  {
    std::printf("Verification passed for all %zu elements.\n", num);
  }

  ////////////////
  // clean up
  ////////////////

  obj->stubble_burn();
  return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
