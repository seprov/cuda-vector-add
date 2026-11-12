"""Generate each pair and add it within one IrGL parallel iteration."""
import cgen
from gg.ast import Module, Kernel, ForAll, CDecl, CBlock
from gg.types import RangeIterator

ast = Module([
    CBlock([cgen.Include("../cuda_generate_values.cuh", system=False)], parse=False),
    Kernel("galois_generate_then_add",
           [("size_t", "len"), ("unsigned int", "seed"), ("int *", "result")], [
        ForAll("i", RangeIterator("len", ty="size_t"), [
            # Use the same device generator as the handwritten CUDA kernels.
            # Only the sum is written to global memory.
            CDecl(("const GeneratedValues", "values", "= cuda_generate_values(i, seed)")),
            CBlock("result[i] = values.first + values.second"),
        ]),
    ]),
])
