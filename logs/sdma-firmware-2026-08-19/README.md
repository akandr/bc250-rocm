**SDMA is not broken on this board. It was the wrong firmware.**

Substituting the navi12 SDMA microcode for the board's own
`cyan_skillfish2_sdma.bin` and `cyan_skillfish2_sdma1.bin` makes every transfer
complete. The tip came from GabriWar, who suggested it in response to the SDMA
notes in this repository.

Both blobs are already shipped by `linux-firmware`, so this is a file
substitution rather than a flash, and it reverts with a copy.

A note for anyone comparing state files, added 26 August. Two digests of this
firmware appear in this repository and they disagree because they digest
different things. `58355d1f7857`, in the `state.txt` of five directories written
on 22 August, is the md5 of the compressed `cyan_skillfish2_sdma.bin.xz`.
`3febefa5396a646fee8f1bd4f453062a`, in the blit recheck of 25 August, is the md5
of the same file decompressed, which is what makes it comparable with
`navi12_sdma.bin.xz` and so what shows the substitution is in place. Both were
recomputed on the board on 26 August and both are right. They are the same SDMA 5.0 format and the same size, and
the version fields are as quoted, `0x34` on the board's original against `0x2c`
on navi12. "Differing only in the ucode version field" was wrong, corrected
26 August: comparing the decompressed blobs byte by byte, 17988 of 33792 differ,
so the payloads are substantially different microcode of the same generation
rather than one program with a version stamp changed. Captured, with the header
fields and the byte comparison, in
[`identity-2026-08-26/`](identity-2026-08-26/). The initramfs carries this firmware, so `dracut -f` is required or the
old blob loads and nothing changes.

**What changed.** Before, a host-to-device copy of 16385 bytes never returned,
in either direction, pinned or pageable, and `HSA_ENABLE_SDMA=0` was required
for every HIP process. After, the full sweep completes:

| size | H2D | D2H |
|---|---|---|
| 4 KiB | 0.27 GB/s | 0.16 GB/s |
| 64 KiB | 2.87 | 2.50 |
| 1 MiB | 6.84 | 8.14 |
| 16 MiB | 30.46 | 30.61 |
| 1 GiB | 47.74 | 47.74 |
| 2 GiB | 47.75 | 47.76 |

Corrected on 25 August. The four smaller rows previously read 0.24/0.14, 2.67/2.36,
6.68/8.50 and 30.51/30.60, which match neither `sweep_sdma_on.txt` in this
directory nor anything else captured in this repository; the values above are the
sweep's. The two largest rows were already exact. The differences run in both
directions, three of the old numbers lower than the sweep and two higher, so this
looks like transcription from a run that was not kept rather than a drift in any
particular direction. Nothing in the write-up turns on these six numbers: the
claim they support is that the sweep completes at every size, which
`ALL SIZES COMPLETED` states directly, and the speed comparisons that matter are
in [`../sdma-sizes-2026-08-19/`](../sdma-sizes-2026-08-19/) with their own data.

`ALL SIZES COMPLETED`, 4 KiB through 2 GiB, zero faults. The one-byte probe that
previously hung at 16385 now loops 481920 copies.

**Correctness is unaffected.** With SDMA enabled the gates return their
established values bit-identically: the 1.5B reads 8.9442 three times, the 8B
reads 9.0975, and the 8.24 GiB model loads without incident.

**It does not make this workload faster, which is worth saying plainly** (though it does help a band of medium transfers; see `../sdma-sizes-2026-08-19/`).
Throughput measured ABBA over three rounds, 8B:

| arm | wall time | pp512 | tg64 |
|---|---|---|---|
| SDMA enabled | 24.91 s | 243.06 | 39.27 |
| SDMA disabled | 24.85 s | 242.92 | 39.24 |

Paired by round, the decode difference is +0.035 t/s, 0.1 percent, with a
standard deviation of the same size. Vulkan is unchanged as well (1.5B pp512
1843.11 against 1842.2 recorded earlier, tg64 210.75 against 211.0; 8B tg64
39.11 against 39.1). GabriWar expected a Vulkan and small-tensor benefit; on
these workloads it is not visible, which is reported as measured rather than as
a disagreement, since neither of those is a small-tensor workload.

Those three Vulkan figures were transcribed from a run that was not kept, the
same gap the sweep table above had. They were re-measured on 25 August on this
board still carrying the navi12 microcode, three repeats each, and they hold:
1.5B pp512 1842.21 to 1843.42, tg64 210.71 to 211.11, 8B tg64 39.05 to 39.11.
The quoted values sit inside all three spreads. The capture is in
[`vk-recheck-2026-08-25/`](vk-recheck-2026-08-25/). This is a reproduction six
days later rather than the original run, so it supports the conclusion without
recovering the lost numbers.

Note also that the blit path the old workaround forced is *faster* for bulk
transfer: about 150 GB/s at 2 GiB against SDMA's 47.8. What SDMA provides is a
separate engine, not more bandwidth. The figure previously read "152 GB/s at
2 GiB from pinned memory". The magnitude is right, 149.59 GB/s in a ten
repeat sweep on 25 August, but the attribution was not: that column is the
ordinary pageable host-to-device copy, and pinned is faster still at 12.96 ms
for 2 GiB, near 166 GB/s. The sweep is in
[`../sdma-sizes-2026-08-19/blit-recheck-2026-08-25/`](../sdma-sizes-2026-08-19/blit-recheck-2026-08-25/).

**One further observation.** With the navi12 firmware, boot logs now show two
`Fence fallback timer expired on ring sdma0` lines. GabriWar's patch header
predicted exactly those, and this repository had previously reported them as
absent. That absence was a property of the cyan firmware, not of the board.

Generated by `scripts/sdma_firmware_ab.sh`; `sweep_sdma_on.txt` is the size
sweep and `dmesg_sdma.txt` the boot lines.
