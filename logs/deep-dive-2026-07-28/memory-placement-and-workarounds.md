# Follow-up: memory placement, and workarounds that do and do not transfer

Continuing the eviction investigation, prompted by ROCm/ROCm#6386 (KFD SVM invalidation triggers a
queue-removal failure and reset on a gfx1103 APU, same mechanism family as the wedge here).

## The standout lead: compute runs in evictable GTT, not VRAM

The wedge is a KFD queue eviction, and a queue only gets evicted if the memory under it is evictable.
On this board that is exactly the case:

```
VRAM total: 512 MB      GTT total: 16384 MB
VRAM used during a GEMM delta: 0 MB   ->  compute allocations land in GTT, not VRAM
```

The BC-250 has a **512 MB VRAM carve-out**; the other ~16 GB is GTT (system memory the GPU maps via
userptr, which is migratable/evictable). So every ROCm allocation is in evictable GTT, and a
`munmap` or SVM invalidation anywhere near it triggers the eviction that the gfx1013 MEC then fails to
preempt. This matches #6386 exactly, where the reporter notes a small UMA carve-out puts all ROCm
allocations in evictable GTT.

`amdgpu.debug_evictions=1` confirms what is being evicted: the wedge is preceded by
`Evicting pid N` / `amdgpu_amdkfd_evict_userptr` followed by `Failed to evict process queues`. So the
eviction that fails is specifically a **userptr eviction** of the GTT-mapped host memory, not
something incidental.

Why compute is in GTT: the driver sets `apu_prefer_gtt = true` for an APU whenever
`real_vram_size < gtt_size` (`amdgpu_ttm.c`). On this board VRAM is 512 MB and GTT is 16 GB, so KFD
placement forces compute into GTT. That gives a concrete, testable fix: make **VRAM >= GTT** so the
flag flips false and compute goes to pinned VRAM. The VRAM split is settable from Linux without a
modified BIOS via [`fanoush/bc250_memcfg`](https://github.com/fanoush/bc250_memcfg), and GTT is capped
with `amdgpu.gttsize`.

**Tested, and it works, partially.** Raising the carve-out alone (`UMA_SIZE 8192`, 8 GB) does nothing,
because 8 GB is still `< 16 GB` GTT, so `apu_prefer_gtt` stays true and compute stays in GTT (VRAM-used
delta 0). But `UMA_SIZE 8192` **plus** `amdgpu.gttsize=4096` (cap GTT at 4 GB, so VRAM 8 GB >= GTT
4 GB) does flip it: during a sustained GEMM, `VRAM_used` is 500-755 MB and `GTT_used` is 12 MB, so
compute now sits in VRAM. The wedge and userptr evictions drop sharply, from near-every heavy run to
roughly 4 of 15, with evictions down to a couple. It is the best mitigation found and it confirms the
mechanism (compute in VRAM is not userptr-evictable). It is not a full fix: HIP's own auxiliary
allocations stay in GTT and still trigger the occasional eviction, so heavy sustained compute still
wedges sometimes. It also costs GTT capacity, which the Vulkan path uses for large models, so it is a
ROCm-compute-only tradeoff. A cheaper proxy, allocating just the compute buffer with `hipHostMalloc`
(pinned) instead of `hipMalloc`, did not help, because it leaves HIP's internal allocations in GTT.
Board restored to 512 MB afterwards.

## A workaround that transfers from gfx1103, and one that does not

ROCm#6386's reporter fixed their gfx1103 freeze with `HSA_USE_SVM=0` + `HSA_ENABLE_SDMA=0` +
`AMD_SERIALIZE_KERNEL=3` plus kernel `amdgpu.cwsr_enable=0` and `amdgpu.ppfeaturemask=0xffff7fff`
(GFXOFF off). Tested here in full on gfx1013: it does **not** transfer. Ten heavy sustained rocBLAS
runs with the complete combination still wedged (4 of the first 4, `cp queue preemption time out`).
The reason is that the gfx1103 failure is MES-based (`MES REMOVE_QUEUE` / `MES might be in
unrecoverable state`), and the workaround leans on MES/power-management behaviour, while gfx1013 has
no MES and wedges on the older nocpsch MEC preemption path. `HSA_USE_SVM=0` alone also did not help
(preemption timeouts persisted), and `AMD_SERIALIZE_KERNEL=3` cannot help this case because the wedge
happens during a single long dispatch, not between kernels.

## Also closed

- Routing HIP compute through the graphics queue (what Mesa does to avoid the broken queue) is not
  available: KFD binds HIP to compute queues; only the userspace graphics driver can reroute.
- `no_queue_eviction_on_vm_fault=1` and `svm_default_granularity` (0 and 18): no effect (the eviction
  is munmap/MMU-notifier driven, not VM-fault driven).

Combining the two partial mitigations does not stack. With the eviction-skip module and VRAM >= GTT
together (compute confirmed in VRAM, VRAM_used ~380 MB), heavy sustained rocBLAS still wedged about 5
of 18. So neither the eviction path nor the memory placement is the whole story: a residual hangs the
dispatch regardless, consistent with the wave-level shader memory-violation seen earlier. Both
mitigations shrink the wedge; neither, nor their combination, removes it.

## Net

No full fix, but a real mechanism-confirmed partial mitigation and a sharpened root cause. The wedge
is a userptr eviction of GTT-resident compute memory that the no-MES gfx1013 MEC cannot preempt.
Compute is in GTT because `apu_prefer_gtt` is set when VRAM < GTT; forcing VRAM >= GTT (`bc250_memcfg
UMA_SIZE 8192` + `amdgpu.gttsize=4096`) moves compute into pinned VRAM and cuts the wedge from
near-every heavy run to about 4 of 15, though HIP's auxiliary GTT allocations keep it from being a full
fix. The gfx1103 SVM/GFXOFF workaround does not transfer (no MES on gfx1013). This is the best
mitigation found and the strongest evidence yet that the wedge is a GTT-eviction problem.
