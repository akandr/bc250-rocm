# The RDNA1 float-activation matrix-vector kernel

`mmvq-rdna1-f32.cu` and its header are the file as it stands after the whole patch series, not after any
one patch: [`../0005-rdna1-mmvq-table-and-sums.patch`](../0005-rdna1-mmvq-table-and-sums.patch) adds it
and [`../0009-rdna1-iq-matvec-shmem-codebook.patch`](../0009-rdna1-iq-matvec-shmem-codebook.patch)
stages the IQ codebooks in shared memory. They are kept here so the kernel can be read on its own; the
patches are what to apply.
