# The zeroed fp16 GEMM: per-tensor instrumentation, 2026-08-14

Evidence for the open defect where the first fp16 attention value-projection of
a graph intermittently returns all zeros while both operands are intact.

The instrumentation prints, per tensor, the element count, sum, sum of squares
and maximum absolute value, plus the source weight's type and how many of its
bytes are non-zero.

| file | what it shows |
|---|---|
| `8b_pure-vcur.txt`, `ds14_pure-vcur.txt` | the same `Vcur-0 MUL_MAT` on consecutive runs: once with `sum=151.01724`, once with `sum=0 sumsq=0 maxabs=0`, while `src0=blk.0.attn_v.weight` reports the identical 8256831 non-zero bytes of 8388608 both times. The weight is intact and the product is empty |
| `conv-vcur.txt` | the ops downstream of it, showing the zeros propagating through RESHAPE and VIEW rather than being introduced later |
| `long_mmq-stats.txt`, `ds14_long_mmq-stats.txt` | the full per-tensor dumps the above were extracted from |

The pattern is that it is the *first* such multiply in a graph execution, one
of 36, and that it follows unrelated work. It is not reproducible in a
standalone program issuing the same shape, and a later cross-check in PyTorch
on the same rocBLAS could not reproduce it either, which points at the calling
pattern rather than at the library or the silicon. The practical avoidance is
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`.

## Downstream effect, recomputed

`long_truth-stats.txt` and `long_mmq-stats.txt` are a clean run and a faulting
run of the same graph, 5790 comparable tensor records each, so they can be
compared position by position. The faulting run contains four zeroed
`Vcur-0 MUL_MAT` records.

Taking the flash attention that follows each zeroed tensor, against the same
position in the clean run:

    divergence after each zeroed tensor:  95, 101, 96, 90 percent
    final logits in the same passes:      48, 48, 48, 48 percent

Two cautions for anyone recomputing this. Compare by graph position rather than
by the first occurrence of a tensor name in the file: the first
`FLASH_ATTN_EXT` in each dump is not the one downstream of the zeroed tensor,
and comparing those two gives a misleading 0 percent. And the top-level README
previously quoted 225 and 44 percent for these; neither follows from these
files, and both were corrected to the numbers above.

## The pattern, counted

Counting `Vcur` `MUL_MAT` records in `long_mmq-stats.txt`: 180 records, five
forward passes of 36 each. The zeroed ones sit at indices 36, 72, 108 and 144,
which is the first record of passes two, three, four and five. Pass one is
clean and no other projection in any pass is affected.

So within a run the defect is not intermittent but deterministic after the
first pass: the first value projection of every pass after the first comes back
empty. Whether a given run shows it at all still varies, which is what the
perplexity arms measure.

Produced by a build carrying the per-tensor statistics hook described in the write-up (`GGML_DEBUG_STATS=1` against a patched `common/debug.cpp`), one line per tensor per run; diff two positionally.
