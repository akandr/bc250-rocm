# The GEMM runs and writes zeros, and the scaling factor is not the cause, 2026-08-23

Generated with the `BC250_TEMP_SENTINEL` and `BC250_ALPHA` instrumentation
already in the tree, on the deterministic baseline: `GGML_CUDA_NO_POOL=1`,
context 2048, ubatch 512, one chunk, fp16 compute. Prediction in
`prediction.txt`, raw lines in `calls.txt`.

## Question one: does the GEMM write zeros, or not write at all?

Those are different defects. A kernel computing the wrong answer belongs to
rocBLAS or the hardware; a dispatch that never lands belongs to the runtime.
Reading zero out of a temporary cannot tell them apart, because a fresh temporary
is already zero.

`BC250_TEMP_SENTINEL` stamps the output temporary with `0x3c` before the GEMM.
As half precision that is about 1.0586, so 524288 elements of stamp sum to
roughly 555000 if never overwritten.

Result: with the sentinel applied to all 144 calls, the zeroed calls still report
`gemm_output_abs_sum=0`.

| call | outcome | sum with sentinel |
|---|---|---|
| 1 | clean | 9361.13 |
| 36 | clean | 1385430 |
| 37 | **zero** | 0 |
| 38 | clean | 23911 |

The stamp is gone, so the GEMM executed and wrote over it. It writes zeros. It is
not a lost dispatch.

That instrument could have said otherwise: had the kernel not written, the sum
would have been about 555000 rather than 0, which is well outside anything else
observed.

## Question two: is the scaling factor zero at those calls?

An earlier dispatch trace had shown alpha and beta being logged as varying
garbage, which was traced to the rocBLAS logging layer misreading a
half-precision scalar. That was on the noisy baseline, so it was worth asking
again on the exact one, using ggml's own probe rather than rocBLAS's logger.

Result: alpha and beta are identical at every one of the 144 calls, including the
zeroed ones. One distinct value across the whole run.

    call 1  (clean) alpha=0.00031995773 beta=0
    call 37 (ZERO)  alpha=0.00031995773 beta=0

So the scalars are not the cause, and the earlier reading that their apparent
variation was a logging artifact is supported.

One caveat on that number rather than on the comparison. The source sets alpha to
1.0 and this probe prints 0.00031995773, so the printed value is itself wrong,
almost certainly the same host-side half-conversion trap this repository has
recorded before. That does not affect the finding, which rests on the values
being equal across calls rather than on what they are, but the absolute figure
should not be quoted.

## Where this leaves the defect

The kernel is dispatched, with identical arguments, with intact operands, with
identical scalars, and it writes zeros. Eliminated so far: the arguments, the
operands, graph capture, the memory pool as a cause, the scaling factors, and a
lost dispatch. What remains is the kernel's own execution on this architecture,
at the first call following a batch that has already run.
