# Blit-path bandwidth, 25 August

The parent page and three other documents said the blit path reaches "124 to
152 GB/s" at 16 MiB. That range appears in no captured data anywhere in this
repository, and it does not describe 16 MiB.

`sweep_sdma_off.txt` is `sdma_probe_repo` with `HSA_ENABLE_SDMA=0`, ten repeats
per size, best of. The relevant rows:

| size | H2D | D2H | H2D pinned |
|---|---|---|---|
| 16 MiB | 110.30 GB/s | 107.56 | 0.13 ms |
| 512 MiB | 150.89 | 146.95 | 3.24 ms |
| 1 GiB | 151.05 | 147.41 | 6.48 ms |
| 2 GiB | 149.59 | 146.16 | 12.96 ms |

So the blit path reaches about 110 GB/s at 16 MiB and about 150 from 512 MiB
upward. The old "124 to 152" is closer to the large-transfer band than to the
size it was attached to. The four times ratio the parent's table reports is
unaffected: 110 against SDMA's 30 GB/s at 16 MiB is the same factor the copy
counts show independently.

The probe reports pinned transfers as a time rather than a rate. At 2 GiB,
12.96 ms is near 166 GB/s, which is why the sibling page's "152 GB/s from
pinned memory" was doubly wrong: 152 is the pageable column, and pinned is
faster than that.

`state.txt` records the kernel, the SDMA firmware checksum and the recovery
parameter this was measured under.
