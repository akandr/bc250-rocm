# The correctness gate, run again, 2026-08-25

`verify.out` is the run: `llama-perplexity` on the 1.5B over eight wikitext
chunks, once with `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` and once with `f16`, on the
board as it now stands.

| arm | recorded in this repository | measured now |
|---|---|---|
| f32 compute type | 8.9442 | **8.9442** |
| f16 compute type | not separately recorded for this model | **8.9442** |

Two things follow.

The gate value reproduces exactly, to four decimals, after a week in which about a
dozen instrumented amdgpu builds went on and off this board and the last was
restored by hand. That is the number this repository quotes more often than any
other, and it had been checked against captured logs rather than produced since
22 August.

The two arms agreeing is also a result, and a confirming one rather than a null.
The fp16 defect is documented as hitting the first fp16 GEMM of each batch on
models that carry F16 value weights, and as never arising on models whose value
weights are quantized. This 1.5B is Q4_K throughout, so it should be on the
unaffected side, and it is: switching the compute type changes nothing at all.
The defect's scope is narrower than "fp16 is broken here", and this is the check
that shows it from the safe side rather than the failing one.

Run in the same spirit as [`../reproduce-verify-2026-08-25/`](../reproduce-verify-2026-08-25/)
and [`../rocm-only-verify-2026-08-25/`](../rocm-only-verify-2026-08-25/): claims
that had only been read were executed instead.
