#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "../cuda_check.cu"

static bool verify_result(const int *device_result, size_t len,
                          unsigned int seed)
{
  // Bound host memory use even when the device arrays are large.
  std::vector<int> chunk(std::min(len, size_t(65536)));
  for (size_t offset = 0; offset < len;)
  {
    const size_t count = std::min(chunk.size(), len - offset);
    CUDA_CHECK(cudaMemcpy(chunk.data(), device_result + offset,
                          count * sizeof(int), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < count; ++j)
    {
      // Independent CPU reference; these products fit in 64 bits.
      const std::uint64_t x = (offset + j) % 10000;
      const int expected = static_cast<int>((seed * x) % 10000 + (seed * x * x) % 10000);
      if (chunk[j] != expected)
      {
        std::fprintf(stderr, "Mismatch at index %zu: got %d, expected %d\n",
                     offset + j, chunk[j], expected);
        return false;
      }
    }
    offset += count;
  }
  return true;
}
