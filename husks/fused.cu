#pragma once

#include "../composites/fused.cuh"
#include "../cuda_check.cu"
#include "../kernels/cuda_generate_then_add.cu"

void FusedComposite::husk()
{
  // allocate device memory
  CUDA_CHECK(cudaMalloc(&d_result_, bytes_));

  CUDA_CHECK(cudaEventCreate(&start_));
  CUDA_CHECK(cudaEventCreate(&stop_));
  float elapsed_ms = 0.0f;

  ///////////////
  // add arrays (generate the inputs here too in fused mode)
  ///////////////

  // start time
  CUDA_CHECK(cudaEventRecord(start_));

  // execute timed code
  cuda_generate_then_add<<<block_count_, threads_per_block_>>>(num_, seed_,
                                                              d_result_);

  // check kernel launch
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop_));
  // call cudaEventSynchronize()
  // Check asynchronous execution errors as well as launch errors.
  CUDA_CHECK(cudaEventSynchronize(stop_));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));

  // print times
  std::printf("Generate + add (fused): %.3f ms\n", elapsed_ms);
}
