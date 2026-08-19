Queue-descriptor sampling during a hanging SDMA copy. **The comparison this
directory was collected for is superseded**; the data is kept because it is real
and because the mistake in reading it is worth preserving.

The intent was to show that the SDMA queue descriptor is frozen during a hang by
contrasting it against sampling during a real compute workload, where descriptors
change substantially between samples. That contrast does not hold up: below the
16384-byte threshold ROCclr uses a blit compute kernel and creates no SDMA queue
at all, so the busy descriptor in the comparison arm was a compute queue. The two
arms never watched the same queue, and no matched control exists on this board,
because no SDMA copy ever succeeds.

What replaced it is in `../../sdma-onebyte-2026-08-18/`: decoding the descriptor
against `struct v10_sdma_mqd` from the kernel headers instead of counting changed
lines. That shows the ring fully configured with its write pointer at zero, which
supports the same conclusion without needing a control.
