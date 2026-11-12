#pragma once

#include <algorithm>
#include <limits>
#include "../composites/galois.cuh"
#include "../cuda_check.cu"
#include "../kernels/generated/galois_vector_add.cuh"

void GaloisComposite::husk()
{
  // allocate device memory: generated inputs stay local to each GPU thread
  CUDA_CHECK(cudaMalloc(&d_result_, bytes_));

  CUDA_CHECK(cudaEventCreate(&start_));
  CUDA_CHECK(cudaEventCreate(&stop_));
  float elapsed_ms = 0.0f;

  // IrGL uses unsigned 32-bit thread IDs and strides, even for size_t indices.
  // Cap the grid so its thread-count multiplication cannot wrap to zero.
  const unsigned int max_blocks =
      std::numeric_limits<unsigned int>::max() / threads_per_block_;
  const unsigned int blocks = std::min(block_count_, max_blocks);

  // start time
  CUDA_CHECK(cudaEventRecord(start_));

  // execute timed code: IrGL coordinates generation and addition in one loop
  galois_generate_then_add<<<blocks, threads_per_block_>>>(num_, seed_, d_result_);

  // check kernel launch and asynchronous execution
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop_));
  CUDA_CHECK(cudaEventSynchronize(stop_));
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_, stop_));

  // print times
  std::printf("Generate + add (Galois): %.3f ms\n", elapsed_ms);
}
