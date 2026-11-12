#pragma once

#include "../composites/split.cuh"
#include "../cuda_check.cu"
#include "../kernels/cuda_generate_arrays.cu"
#include "../kernels/cuda_vector_add.cu"

void SplitComposite::husk()
{
  // Allow this composite to run again without keeping the previous allocations.
  stubble_burn();

  // allocate device memory
  CUDA_CHECK(cudaMalloc(&d_vec1_, bytes_));
  CUDA_CHECK(cudaMalloc(&d_vec2_, bytes_));
  CUDA_CHECK(cudaMalloc(&d_result_, bytes_));

  CUDA_CHECK(cudaEventCreate(&start_));
  CUDA_CHECK(cudaEventCreate(&stop_));
  float elapsed_ms = 0.0f;

  //////////////////
  // generate arrays
  //////////////////

  // start time
  CUDA_CHECK(cudaEventRecord(start_));

  // execute timed code
  cuda_generate_arrays<<<block_count_, threads_per_block_>>>(num_, seed_,
                                                            d_vec1_, d_vec2_);
  // check kernel launch
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop_));
  // call cudaEventSynchronize()
  // Check asynchronous execution errors as well as launch errors.
  CUDA_CHECK(cudaEventSynchronize(stop_));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));

  // print times
  std::printf("Generate: %.3f ms\n", elapsed_ms);

  ///////////////
  // add arrays
  ///////////////

  // start time
  CUDA_CHECK(cudaEventRecord(start_));

  // execute timed code
  cuda_vector_add<<<block_count_, threads_per_block_>>>(num_, d_vec1_, d_vec2_,
                                                       d_result_);

  // check kernel launch
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop_));
  // call cudaEventSynchronize()
  CUDA_CHECK(cudaEventSynchronize(stop_));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));

  // print times
  std::printf("Add: %.3f ms\n", elapsed_ms);
}
