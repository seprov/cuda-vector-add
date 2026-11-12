#pragma once

#include <cstddef>
#include <cuda_runtime.h>

struct GeneratedValues
{
  int first;
  int second;
};

// Generate one pair of deterministic test values in the calling GPU thread.
static __device__ GeneratedValues cuda_generate_values(std::size_t index,
                                                       unsigned int seed)
{
  // Reduce before multiplying so even very large indices cannot overflow.
  const unsigned int x = static_cast<unsigned int>(index % 10000);
  const unsigned int first = ((seed % 10000) * x) % 10000;
  return {static_cast<int>(first), static_cast<int>((first * x) % 10000)};
}
