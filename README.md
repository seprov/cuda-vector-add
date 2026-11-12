# CUDA integer vector addition

CUDA integer vector addition with deterministic input generation and a CPU
correctness check of every result. Requires the CUDA Toolkit and an NVIDIA GPU.

## Compile for an RTX 3090 (compute capability 8.6):

```sh
nvcc -O3 -std=c++17 -arch=sm_86 cuda_vector_add.cu -o cuda_vector_add
./cuda_vector_add 1025
```

On Windows, use `-o cuda_vector_add.exe` and run `.\cuda_vector_add.exe 1025`.
Adjust `-arch` for other GPUs. The length must be a positive decimal integer.

The optional second argument selects `separate` (the default) or `fused`:

```sh
./cuda_vector_add 1025 separate
./cuda_vector_add 1025 fused
```

Separate mode generates two input arrays, then adds them with a second kernel.
Fused mode generates each pair of input values and adds them in one kernel,
writing only the result array. Both modes use the same per-element `__device__`
helper, `cuda_generate_values`, which returns two scalar values to its calling
GPU thread. The `__global__` kernels write their outputs through device pointers.

The program reports CUDA event timings for generation and addition (one combined
timing in fused mode), then copies
the result back in small chunks and checks every element against a CPU reference.
Verification is outside the timed intervals. These are single-run timings, not a
warmed-up benchmark. The generated values preserve the original modular formulas
without signed overflow; they are test data, not statistically random samples.

Separate mode's three device arrays require `12 * length` bytes: one billion
elements need 12 GB, and ten billion need 120 GB, before CUDA overhead. Fused mode
only allocates the result array, requiring `4 * length` bytes. Requests exceeding
available device memory are rejected. Host verification uses at most 256 KiB.

## Composites

`main` selects a `FusedComposite` or `SplitComposite`, then calls:

```cpp
obj->husk();           // Allocate memory, launch the kernels, and report timings.
obj->verify_result();  // Check the result on the CPU while device memory is live.
obj->stubble_burn();   // Release device memory and timing events.
```

Class declarations live in `composites/`, with method implementations in
`husks/` and `stubble_burning/`. Both composites share the CPU check in
`verification/`. Each object owns its CUDA resources and cannot be copied.
Cleanup is safe to call more than once, and the destructor also calls it when
the object leaves scope. Calling `husk()` again replaces the previous run.

The build command above still compiles only `cuda_vector_add.cu`. That file
includes the implementation files; they do not need to be compiled separately.

The lifecycle test covers repeated runs, repeated cleanup, destructor cleanup,
and verification order for both composites with several seeds:

```sh
nvcc -O3 -std=c++17 -arch=sm_86 tests/composites_test.cu -o /tmp/composites_test
/tmp/composites_test
```

It deliberately attempts verification before running and after cleanup, so
twelve verification-order error messages are expected.

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

To profile the fused kernel:

```sh
ncu --set basic --kernel-name cuda_generate_then_add ./cuda_vector_add 1000000 fused
```

The fused kernel performs both generation and addition, so compare it with both
separate kernels together when evaluating the total work.
