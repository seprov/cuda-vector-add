# CUDA integer vector addition

CUDA integer vector addition with deterministic input generation and a CPU
correctness check of every result. Requires the CUDA Toolkit and an NVIDIA GPU.

## Compile for an RTX 3090 (compute capability 8.6):

```sh
nvcc -O3 -std=c++17 -arch=sm_86 cuda_vector_add.cu -o cuda_vector_add
./cuda_vector_add 1025
```

On Windows, use `-o cuda_vector_add.exe` and run `.\cuda_vector_add.exe 1025`.
Adjust `-arch` for other GPUs. The argument must be a positive decimal integer.

The program reports CUDA event timings for generation and addition, then copies
the result back in small chunks and checks every element against a CPU reference.
Verification is outside the timed intervals. These are single-run timings, not a
warmed-up benchmark. The generated values preserve the original modular formulas
without signed overflow; they are test data, not statistically random samples.

The three device arrays require `12 * length` bytes: one billion elements need
12 GB, and ten billion need 120 GB, before CUDA overhead. Requests exceeding
available device memory are rejected. Host verification uses at most 256 KiB.

## For a memory check, if Compute Sanitizer is installed:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 ./cuda_vector_add 1025
```

## For Nsight metrics,

```sh
export PATH="$HOME/research/pysr-gpu-backend/.tools/nsight/bin:$PATH"
```

And then either 

```sh
ncu --set full --kernel-name cuda_vector_add \
    -o vector-add ./cuda_vector_add 1000000
```

or

```sh
ncu --set basic --kernel-name cuda_vector_add ./cuda_vector_add 1000000
```