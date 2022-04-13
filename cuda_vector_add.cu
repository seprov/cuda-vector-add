#include <cstdio>
#include <algorithm>
#include <ctime>
#include <cuda.h>

//static const int ThreadsPerBlock = 512;
#define THREADS_PER_BLOCK 512

// --use_fast_math
// 1_000_000_000 works, 10_000_000_000 does not work

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    //e = cudaGetLastError();
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

static __global__ void cuda_vector_add(const long long * len, int* const vec1, int* const vec2, int* rslt)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < *len)
    rslt[i] = vec1[i] + vec2[i];
}

static __global__ void cuda_generate_random_arrays(const long long * len, const int * seed, int* const vec1, int* const vec2)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < *len) {
    vec1[i] = (*seed * i) % 10000;
    vec2[i] = (*seed * i * i) % 10000;
  }
}

void printCudaProperties() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}

int main(int argc, char* argv[])
{
  ////////////
  // set up
  ////////////
  printf("CUDA Vector Add v1.0\n");
  //printCudaProperties();
  cudaSetDevice(0);
  // check command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s length_of_vectors\n", argv[0]); exit(-1);}
  const long long num = atoll(argv[1]);
  printf("length of vectors: %llu\n", num);
  const int seed = 111111;
  printf("seed: %i\n", seed);
  long long unsigned blockCount = (num + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;

  // allocate device memory
  int* d_vec1 = new int [num];
  int* d_vec2 = new int [num];
  int* d_rslt = new int [num];
  const long long * d_num = &num;
  const int* d_seed = &seed;
  if (cudaSuccess != cudaMalloc((void **)&d_vec1, sizeof(int)*num)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);} 
  if (cudaSuccess != cudaMalloc((void **)&d_vec2, sizeof(int)*num)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);} 
  if (cudaSuccess != cudaMalloc((void **)&d_rslt, sizeof(int)*num)) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&d_num, sizeof(int))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  if (cudaSuccess != cudaMalloc((void **)&d_seed, sizeof(int))) {fprintf(stderr, "ERROR: could not allocate memory\n"); exit(-1);}
  
  //////////////////
  // generate arrays
  //////////////////
  printf("generating arrays\n");

  // start time
  clock_t t = clock();

  // execute timed code
  cuda_generate_random_arrays<<<blockCount, THREADS_PER_BLOCK>>>(d_num, d_seed, d_vec1, d_vec2);

  // call cudaDeviceSynchronize()
  cudaDeviceSynchronize();
  t = clock() - t;

  // print times
  printf ("It took %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);

  // call CheckCuda()
  CheckCuda();

  ///////////////
  // add arrays
  ///////////////
  printf("adding arrays\n");

  // start time
  t = clock();

  // execute timed code
  cuda_vector_add<<<blockCount, THREADS_PER_BLOCK>>>(d_num, d_vec1, d_vec2, d_rslt);

  // call cudaDeviceSynchronize()
  cudaDeviceSynchronize();
  t = clock() - t;

  // print times
  printf ("It took %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);

  ////////////////
  // clean up
  ////////////////

  // copy from GPU to CPU
  // int* rslt = new int [num];
  // if (cudaSuccess != cudaMemcpy(&rslt, d_rslt, sizeof(int)*num, cudaMemcpyDeviceToHost)) {fprintf(stderr, "ERROR: copying from device failed\n"); exit(-1);}

  // clean up
  cudaFree(d_vec1);
  cudaFree(d_vec2);
  cudaFree(d_rslt);
 
  return 0;
}

