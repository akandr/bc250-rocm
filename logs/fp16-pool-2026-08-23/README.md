# The memory pool causes the non-determinism, not the defect, 2026-08-23

Generated with `BC250_TEMP` counting zeroed GEMM outputs directly, five runs per
arm, on one perplexity chunk at context 2048, ubatch 512, fp16 compute. The
prediction was written first and is in `prediction.txt`; raw counts in `runs.txt`.

## Why the zero count and not perplexity

This repository already records that bypassing the pool "changes nothing", judged
on perplexity. Perplexity is a coarse instrument here: it moves for many reasons
and compresses the thing of interest into one number. Counting the zeroed GEMMs
is far sharper, and it turned out to matter.

## Result

| arm | zeros per run | positions | perplexity |
|---|---|---|---|
| pool enabled | 3, 3, 4, 2, 2 | 37/73/109, 73/74/109, 73/74/109/110, 73/109, 73/109 | 14.1344, 14.6752, 16.9875, 11.7443, 11.7443 |
| `GGML_CUDA_NO_POOL=1` | 3, 3, 3, 3, 3 | 37/73/109 every time | 14.1344 every time |
| single batch, ub 2048 | 0, 0, 0, 0, 0 | none | 7.2768 every time |

With the pool disabled the defect is perfectly deterministic: exactly three
zeros, exactly at calls 37, 73 and 109, which are the first fp16 GEMM of the
second, third and fourth batches, and an identical perplexity five times over.

With the pool enabled the count wanders between two and four, the positions
drift onto the call after a batch start as well as the start itself, and the
perplexity takes four different values across five runs.

## What this changes

The non-determinism that has characterised this defect since it was found is not
a property of the defect. It is the memory pool moving things around on top of a
defect that is otherwise exact. Underneath, the batch-boundary model holds
precisely: one zeroed GEMM per batch after the first, no more and no fewer.

The pool is still cleared as a cause. Disabling it does not remove the zeros, it
only stops them wandering.

## A correction to what this repository claimed earlier today

Before this run the batch-boundary model had been recorded here as "exactly one zero per
batch after the first", on the strength of several runs that all showed three at
ubatch 512. That was luck: with the pool enabled the count varies, and one of the
runs in this very session showed seven. The model is correct, but only with the
pool disabled; with it enabled the count is a distribution around it.

The ubatch prediction test that appeared to confirm the model exactly, none, one,
three and eight zeros for one, two, four and eight batches, was a single run per
condition of a quantity that varies. Its agreement was real but weaker evidence
than it looked.

## What survives untouched

The single-batch workaround. Five runs at ubatch 2048 give no zeros at all and an
identical 7.2768, against an f32 reference of 7.2672. That claim is now on five
samples rather than three and is the most robust thing here.

## An arm that could not be run

`GGML_CUDA_POOL_NOREUSE=1` holds every allocation forever, which exhausts memory
on this board: the run aborted after 96 of the expected 144 calls with no
perplexity. It is recorded as untestable rather than as a result.
