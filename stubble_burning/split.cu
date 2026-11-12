#pragma once

#include "../composites/split.cuh"
#include "../cuda_check.cu"

void SplitComposite::stubble_burn()
{
  // Clear each handle so explicit cleanup and destructor cleanup can both run.
  if (start_ != nullptr)
  {
    CUDA_CHECK(cudaEventDestroy(start_));
    start_ = nullptr;
  }
  if (stop_ != nullptr)
  {
    CUDA_CHECK(cudaEventDestroy(stop_));
    stop_ = nullptr;
  }
  if (d_vec1_ != nullptr)
  {
    CUDA_CHECK(cudaFree(d_vec1_));
    d_vec1_ = nullptr;
  }
  if (d_vec2_ != nullptr)
  {
    CUDA_CHECK(cudaFree(d_vec2_));
    d_vec2_ = nullptr;
  }
  if (d_result_ != nullptr)
  {
    CUDA_CHECK(cudaFree(d_result_));
    d_result_ = nullptr;
  }
}
