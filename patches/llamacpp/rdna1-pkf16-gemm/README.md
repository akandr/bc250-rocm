# The RDNA1 packed-fp16 prefill GEMM

`mmf16-rdna1.cu` and its header are the file as it stands after the whole patch series, not after any one
patch: [`../0008-rdna1-pkf16-prefill-gemm.patch`](../0008-rdna1-pkf16-prefill-gemm.patch) adds it and
[`../0010-rdna1-pkf16-halve-column-tile.patch`](../0010-rdna1-pkf16-halve-column-tile.patch) changes the
tile from 128x128 to 128x64,
[`../0011-rdna1-pkf16-q8_0.patch`](../0011-rdna1-pkf16-q8_0.patch) brings q8_0 onto it and
[`../0012-rdna1-pkf16-expert-path.patch`](../0012-rdna1-pkf16-expert-path.patch) adds the
mixture-of-experts path and
[`../0013-rdna1-pkf16-admit-128-token-batches.patch`](../0013-rdna1-pkf16-admit-128-token-batches.patch)
lowers the batch it will take. They are kept here so the kernel can be read on its own; the patches are
what to apply.
