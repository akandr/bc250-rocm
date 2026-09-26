# Kernel 7.2.5 with the board patches, 2026-09-17

Fedora 44's kernel 7.2.5-200 was tested because the 7.1.y branch this repository measures on is missing
amdkfd fixes that 7.2.y carries, including the TLB flush added after MES queue eviction, and because the
board's remaining open defects are all kernel-side. The question was whether any of that removes the need
for the runlist-rebuild flush.

**It does not.** The board behaves exactly as on 7.1.8.

## Building it

All four board patches applied to the vanilla 7.2.5 source with no hand editing:

- [`scripts/apply_runlist_flush.py`](../../scripts/apply_runlist_flush.py) and
  [`scripts/apply_svmflush_generic.py`](../../scripts/apply_svmflush_generic.py) applied clean.
- The 40-CU unlock and the `bc250_flush_pasid_kiq` parameter were extracted as diffs against vanilla 7.1.8
  and applied to 7.2.5 with offsets of 2 and -4 lines, no failed hunks.
- [`scripts/build_patched_amdgpu.sh`](../../scripts/build_patched_amdgpu.sh) built and installed the module
  and rebuilt the initramfs, with `SRC` and `KREL` pointed at the new tree.

After booting: `uname -r` 7.2.5-200.fc44, all four `bc250_*` parameters present, `simd_count 80` (40 CU),
no failed units (`environment.txt`).

## Results

Gates, ROCm backend, default compute type (`gates.txt`):

| gate | kernel 7.2.5 | kernel 7.1.8 |
|---|---|---|
| qwen2.5-1.5B, ctx 4096, 8 chunks | 8.9442 | 8.9442 |
| qwen3-8B, ctx 2048, 2 chunks | 9.1117 | 9.1117 |
| qwen3-14B, ctx 2048, 2 chunks | 7.7645 | 7.7645 |

Throughput, medians of three (`*_pp.jsonl`, `*_tg.jsonl`):

| | kernel 7.2.5 | kernel 7.1.8 |
|---|---|---|
| qwen2.5-1.5B pp512 / tg64 | 793.3 / 119.7 | 792.9 / 117.5 |
| qwen3-8B pp512 / tg64 | 244.2 / 39.4 | 243.6 / 38.9 |

Allocation-churn A/B/A (`churn_aba.txt`): with `bc250_flush_by_runlist=3`, two runs clean and no kernel
fault lines; at 1, both runs die with `Memory access fault by GPU` and the journal gains 20 fault lines;
back at 3, clean again with no new lines.

## Reading

Five kernel releases now measure the same with this patch set, 6.18.9 through 7.2.5, which extends the
kernel-independence finding instead of changing it. The amdkfd difference between the branches is visible
in the source, six `kfd_flush_tlb` call sites in 7.2.5's queue manager against four in 7.1.8, and it makes
no difference here: the flush this board needs is the runlist rebuild, and nothing upstream provides it.

The gates being identical across a kernel change is also a useful control on the gates themselves.
