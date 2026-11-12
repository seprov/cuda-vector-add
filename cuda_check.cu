#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static void check_cuda(cudaError_t error, const char *call, int line)
{
  if (error != cudaSuccess)
  {
    std::fprintf(stderr, "CUDA error at line %d in %s: %s\n",
                 line, call, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)
