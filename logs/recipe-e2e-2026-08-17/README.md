# End-to-end verification of the documented patch recipe, 2026-08-17

The recipe in the top-level README was written from a tree that had been
patched incrementally over weeks. This checks it the way a reader would meet
it: restore a kernel tree to pristine, run only the two documented scripts,
build, install, boot, and validate.

Tree: Fedora 6.18.16, restored from the `.pre-runlist` backups so no hook was
present in any file. Sequence: `apply_runlist_flush.py <amdkfd>` then
`apply_svmflush_generic.py <amdkfd>`, nothing else.

Resulting hook set, identical to the modules that booted clean previously:

    kfd_chardev.c              2
    kfd_device_queue_manager.c 1
    kfd_device_queue_manager.h 1
    kfd_svm.c                  2

Booted with `bc250_flush_by_runlist=3`: 40 CU (`simd_count 80`), map bit
present in the loaded module. Validation battery, all in `inv47.log`:

    compute probe 4096/16384/32768 blocks   ALL CORRECT
    SGEMM N=256 through 4096                0 wrong at every size
    perplexity gate                         8.9442, matching the reference
    sustained N=4096 x50                    0 wrong
    dmesg faults                            0

This also makes 6.18.16 the third kernel on which the working configuration has
been measured as equivalent, after 6.18.9 and 7.1.5.

Two defects in `apply_runlist_flush.py` were found by this exercise and fixed
before the run above: it ignored its directory argument and patched a hardcoded
path, and its `kfd_chardev.c` anchor matched only kernels where `kfd_flush_tlb`
takes one argument, so it failed silently on 6.x.

Collected by running the write-up's recipe end to end on a fresh boot and recording each step's output.
