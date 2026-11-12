# CUDA integer vector addition

CUDA integer vector addition with deterministic input generation and a CPU
correctness check of every result. Requires the CUDA Toolkit, an NVIDIA GPU,
and an external Galois checkout for the generated kernel's GPU helper header.

## Compile for an RTX 3090 (compute capability 8.6):

```sh
export GALOIS_ROOT="$HOME/research/Galois"
nvcc -O3 -std=c++17 -arch=sm_86 -I"$GALOIS_ROOT/libgpu/include" \
  cuda_vector_add.cu -o cuda_vector_add
./cuda_vector_add 1025
```

On Windows, use `-o cuda_vector_add.exe` and run `.\cuda_vector_add.exe 1025`.
Adjust `-arch` for other GPUs. The length must be a positive decimal integer.

The optional second argument selects `separate` (the default), `fused`, or `galois`:

```sh
./cuda_vector_add 1025 separate
./cuda_vector_add 1025 fused
./cuda_vector_add 1025 galois
```

Separate mode generates two input arrays, then adds them with a second kernel.
Fused mode generates each pair of input values and adds them in one kernel,
writing only the result array. All modes use the same per-element `__device__`
helper, `cuda_generate_values`, which returns two scalar values to its calling
GPU thread. The `__global__` kernels write their outputs through device pointers.
Galois mode also generates and adds in one kernel. Its IrGL `ForAll` calls
`cuda_generate_values` and adds the returned scalars within the same iteration,
writing only the result. There are no intermediate input arrays or separate
generation kernel in this mode.

The program reports CUDA event timings for generation and addition (one combined
timing in fused and Galois modes), then copies
the result back in small chunks and checks every element against a CPU reference.
Verification is outside the timed intervals. These are single-run timings, not a
warmed-up benchmark. The generated values preserve the original modular formulas
without signed overflow; they are test data, not statistically random samples.

Separate mode's three device arrays require `12 * length` bytes: one billion
elements need 12 GB, and ten billion need 120 GB, before CUDA overhead. Fused and
Galois modes only allocate the result array, requiring `4 * length` bytes. Requests exceeding
available device memory are rejected. Host verification uses at most 256 KiB.

## Composites

`main` selects a `FusedComposite`, `SplitComposite`, or `GaloisComposite`, then calls:

```cpp
obj->husk();           // Allocate memory, launch the kernels, and report timings.
obj->verify_result();  // Check the result on the CPU while device memory is live.
obj->stubble_burn();   // Release device memory and timing events.
```

Class declarations live in `composites/`, with method implementations in
`husks/` and `stubble_burning/`. All composites share the CPU check in
`verification/`. Each object owns its CUDA resources and cannot be copied.
Cleanup is safe to call more than once, and the destructor also calls it when
the object leaves scope. The intended lifecycle is one run followed by
verification and cleanup. Call `stubble_burn()` before starting another run on
the same object; `husk()` does not perform cleanup at the beginning.

`GaloisComposite` inherits the fused composite's single-result buffer layout and
cleanup. Its own `husk()` allocates that buffer and times one IrGL-generated
kernel covering both generation and addition.

The build command above still compiles only `cuda_vector_add.cu`. That file
includes the implementation files; they do not need to be compiled separately.

The lifecycle test covers reuse after cleanup, repeated cleanup, destructor
cleanup, and verification order for all composites with several seeds:

```sh
nvcc -O3 -std=c++17 -arch=sm_86 -I"$GALOIS_ROOT/libgpu/include" \
  tests/composites_test.cu -o /tmp/composites_test
/tmp/composites_test
```

It deliberately attempts verification before running and after cleanup, so
eighteen verification-order error messages are expected.

## Galois dependencies and generation

Use external checkouts while experimenting with the compiler and runtime:

- `Galois` supplies `libgpu/include/ggcuda.h`, the thread-index and stride helpers.
- `galois-compiler` is a build tool that translates the IrGL `ForAll` in
  `kernels/galois_vector_add.py` into a CUDA kernel. It does not run on the GPU.

On this WSL machine the copies are `/home/skapden/research/Galois` and
`/home/skapden/research/galois-compiler`. Set `GALOIS_ROOT` to the appropriate
path on other machines. The generated files in `kernels/generated/` are kept
with the project, so ordinary builds need only the Galois header. Regenerate
after editing the IrGL source or intentionally changing compiler versions:

```sh
export GALOIS_COMPILER_ROOT="$HOME/research/galois-compiler"
export IRGL_PYTHON="$HOME/research/pysr-gpu-backend/.tools/irgl-python/bin/python"
"$IRGL_PYTHON" tools/generate_galois.py --compiler "$GALOIS_COMPILER_ROOT"
```

`IRGL_PYTHON` is the existing Python environment with this fork's dependencies.
On a new machine, follow the compiler fork's Python 3 installation instructions
and use that environment's Python interpreter.

The adapter removes unused graph-application boilerplate; it preserves the
compiler's generation-and-addition loop and GPU thread mapping. The raw output and adaptation diff
are saved beside the generated header for inspection. This flat vector loop
does not need the full Galois graph/worklist runtime or its legacy dependencies.
It uses a regular grid-stride loop, without nested or irregular scheduling.

Git submodules can pin both forks to exact commits once the integration is
stable. External paths currently use whatever code is present in those copies.

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
ncu --set full --kernel-name cuda_vector_add split \
    -o vector-add ./cuda_vector_add 1000000
```

or

```sh
ncu --set basic --kernel-name cuda_vector_add ./cuda_vector_add split 1000000
```

To profile the fused kernel:

```sh
ncu --set basic --kernel-name cuda_generate_then_add ./cuda_vector_add 1000000 fused
```

The fused kernel performs both generation and addition, so compare it with both
separate kernels together when evaluating the total work.

To profile the IrGL-generated generation-and-addition kernel:

```sh
ncu --set basic --kernel-name galois_generate_then_add ./cuda_vector_add 1000000 galois
```
