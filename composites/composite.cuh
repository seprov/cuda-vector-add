#pragma once

#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include "../verification/verify_vector_add.cu"

// A composite owns the device memory and events for one way of doing the work.
class Composite
{
public:
  Composite(std::size_t num, unsigned int seed, unsigned int block_count,
            unsigned int threads_per_block)
      : num_(num), bytes_(num * sizeof(int)), seed_(seed),
        block_count_(block_count), threads_per_block_(threads_per_block)
  {
  }

  virtual ~Composite() = default;

  // Copying would give two objects ownership of the same CUDA resources.
  Composite(const Composite &) = delete;
  Composite &operator=(const Composite &) = delete;

  virtual void husk() = 0;
  virtual void stubble_burn() = 0;

  bool verify_result() const
  {
    if (d_result_ == nullptr)
    {
      std::fprintf(stderr, "ERROR: call husk() before verification, and "
                           "verify before stubble_burn()\n");
      return false;
    }
    return ::verify_result(d_result_, num_, seed_);
  }

protected:
  const std::size_t num_;
  const std::size_t bytes_;
  const unsigned int seed_;
  const unsigned int block_count_;
  const unsigned int threads_per_block_;

  int *d_result_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};
