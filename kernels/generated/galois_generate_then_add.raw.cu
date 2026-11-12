/*  -*- mode: c++ -*-  */
#include "gg.h"
#include "ggcuda.h"

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);
#define TB_SIZE 256
const char *GGC_OPTIONS = "cuda.worklist_type=basic $ cuda.graph_type=basic $ cuda.use_worklist_slots=True $ unroll=[] $ cc_disable=set() $ instrument=set() $ hacks=set() $ read_props=None $ write_props=None $ outline_iterate=False $ outline_iterate_gb=False $ np=False $ coop_conv=False $ np_schedulers={'tb', 'fg', 'wp'} $ np_factor=1 $ parcomb=False $ quiet_cgen=True $ ignore_nested_errors=False $ retry_backoff=False $ backoff_blocking_factor=4 $ instrument_mode=None";
#include "../cuda_generate_values.cuh"
__global__ void galois_generate_then_add(size_t len, unsigned int seed, int * result)
{
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  size_t i_end;
  // FP: "1 -> 2;
  i_end = len;
  for (size_t i = 0 + tid; i < i_end; i += nthreads)
  {
    const GeneratedValues values = cuda_generate_values(i, seed);
    result[i] = values.first + values.second;
  }
  // FP: "5 -> 6;
}