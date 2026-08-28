# The zeroed GEMM has good operands and identical arguments, 2026-08-23

Generated with instrumentation already in the tree, on one perplexity chunk at
context 2048, ubatch 512, fp16 compute. Raw lines in `calls.txt`, which prints
them under the markers `BC250PTR`, `BC250DEV` and `BC250TEMP`.

Two notes on reading it, added 26 August. This line used to name the switches as
`BC250_PTR`, `BC250_DEVCHECK` and `BC250_TEMP`. Only the last of those exists as
an environment variable in any shipped script; the other two appear nowhere, and
the capture prints the undescored marker names above. The instrumentation was in
the working tree rather than in a harness, and that tree is not shipped, so the
markers are what can be checked. And the call indices below, 1, 36, 37, 38 and
73, are not in `calls.txt`: it records tensor names in order, and the indices are
reconstructed from the thirty-six calls per batch established in
[`../fp16-pool-2026-08-23/`](../fp16-pool-2026-08-23/). They are consistent with
it, and they are a reading of the capture rather than a field in it.

## What was asked

The defect is one zeroed fp16 GEMM per evaluated batch after the first. The
question is what differs between the call that works and the call that does not.
The natural candidates are the arguments and the operands.

## Neither differs

The zeroed calls and the clean one are the same tensor, `Vcur-0`, the layer-0
value projection, and their arguments are byte-identical:

    call 1  (clean) src0=0x7fb0a4be0000(al=131072) src1=0x7fb003200000(al=2097152) dst=0x7fb02f003000(al=4096) ne11=512
    call 37 (ZERO)  src0=0x7fb0a4be0000(al=131072) src1=0x7fb003200000(al=2097152) dst=0x7fb02f003000(al=4096) ne11=512
    call 73 (ZERO)  src0=0x7fb0a4be0000(al=131072) src1=0x7fb003200000(al=2097152) dst=0x7fb02f003000(al=4096) ne11=512

Same pointers, same alignments, same shape.

And the operands are intact as the shader cores themselves see them, summed on
device rather than read back through the host:

| call | outcome | weight abs sum | activations abs sum |
|---|---|---|---|
| 1 | correct | 84296.1 | 16499.4 |
| 36 | correct | 95834.8 | 3126320 |
| 37 | **zero** | 84296.1 | 17028.6 |
| 38 | correct | 88083.7 | 32439.6 |
| 73 | **zero** | 84296.1 | 17059.3 |

The weight sum at the zeroed calls is identical to the clean call's, to six
figures, and the activations are healthy and of the expected magnitude. Good
inputs, and the GEMM leaves exactly zero in its output temporary.

## The instrument could have said otherwise

Two checks, because a negative from an instrument is worth nothing unless it
could have produced a positive. The device-side sums are non-zero for every call
including the zeroed ones, so the probe reports non-zero when non-zero is there.
And the defect survives the instrumentation: perplexity is 14.1344 with the
probes on, the same value measured without them, with the zeros at the same
positions 37, 73 and 109.

## What is left

Not the arguments, not the operands, not graph capture, not the batch size beyond
its effect on batch count. The same call with the same pointers and the same
input data returns a correct result in the first batch and nothing in later ones.

That is a statement about execution rather than about data, and it narrows to
whatever the batch boundary changes underneath an otherwise identical dispatch.
The output temporary's address is the obvious next thing to record, since
`BC250_TEMP` currently reports only its contents.
