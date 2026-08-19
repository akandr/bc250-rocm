# Deep-dive round, 2026-07-28

Follow-up investigation prompted by recent ROCm/ROCm#6313 activity, pushing on four threads.
All on kernel 6.18.9, 40 CU, flush stock (true), `HSA_ENABLE_SDMA=0`. Board power-cycled between
runs for fresh state.

## 1. The compute-queue fault, decoded (new)

Reproducing the fault under `compute_probe` and reading the full amdgpu VM-fault block from one
captured instance (`l2_fault_decode.log`) names the faulting client:

```
[gfxhub] page fault vmid:8 ... from client 0x1b (UTCL2)
GCVM_L2_PROTECTION_FAULT_STATUS:0x00801031
   Faulty UTCL2 client ID: TCP (0x8)
   WALKER_ERROR: 0x0   MAPPING_ERROR: 0x0   PERMISSION_FAULTS: 0x3   RW: 0x0 (read)
```

- **Client is TCP** (Texture Cache per Pipe), i.e. the shader's vector-L1 / vector-memory path. The
  faulting access is a shader vector memory op, not something incidental.
- **The page is mapped and the walk succeeds** (`MAPPING_ERROR=0`, `WALKER_ERROR=0`); the access is
  rejected on **permission** (`PERMISSION_FAULTS=0x3`), on a **read** (`RW=0`). So this is a
  permission rejection on a mapped page, not a missing one.

That is a single decode; what follows is interpretation, not measurement. A permission fault on a
mapped page, on a read, matches the known pattern of a buffer unmapped while a shader still references
it (a use-after-free on the GPU side). That is exactly the runtime `munmap` churn and queue eviction
of Observation 2 seen from the memory controller: the same eviction that usually times out the MEC
preemption and wedges the queue can instead let an in-flight read land on a just-revoked page and
fault (the fault address is in the process's SVM range). A stale translation from the PASID flush
(the correctness-defect mechanism) could also leave a wrong permission, so one trace does not cleanly
separate the two. External reports of the same `GCVM_L2_PROTECTION_FAULT` / TCP permission-fault class
on other AMD parts attribute it to buffer use-after-free, which is the same shape. Offered as a
hypothesis, from one board. The same `GCVM_L2_PROTECTION_FAULT` is
the fault class other #6313 participants report from an image-bandwidth test.

## 2. A minimal read-only reproducer

`../../patches/bw_probe.cpp` is a bare HIP streaming-read kernel (no arithmetic, no rocBLAS). On a
fresh boot it completes at 16-256 MB, aborts at 1 GB (`HSA_STATUS_ERROR_INVALID_PACKET_FORMAT`), and
wedges at 2 GB. A pure vector-read workload reproduces the size-dependent defect, stressing exactly
the TCP path above. (An earlier version had an out-of-bounds output write that hung regardless of
size; that bug is fixed in the committed version.)

ROCm OpenCL (for clpeak) was a dead end: `rocm-opencl` 6.4.2 is installed but enumerates zero
OpenCL platforms for gfx1013, so clpeak cannot run on the compute queue.

## 3. PyTorch (mainstream framework) hits the same wall

`pytorch_native_gfx1013_matmul.log`, `torch_sweep.py`:

- The official `torch 2.9.1+rocm6.4` wheel detects the board ("AMD BC-250", `cuda.is_available()`
  True) but ships **no gfx1013 and no gfx1010** rocBLAS kernels, so a matmul fails immediately
  (`Illegal seek for GPU arch : gfx1013`). Out of the box, the wheel cannot run a single op here.
- Grafting a native gfx1013 rocBLAS into the wheel makes it run: single matmuls are correct up to
  N=4096 (rel err ~1e-6), but a sustained loop (N=4096 x50) wedges the compute queue (`queue
  evicted` / preemption timeout). So the defect bites PyTorch exactly as it bites rocBLAS and the
  probes; a real training or inference workload (many sustained matmuls) wedges.

Also noted: Fedora's system rocBLAS "supports gfx1013" only by **symlinking gfx1013 to gfx1010**
kernels (`rocblas_gfx1013_symlinks.log`), i.e. the override approach baked into the package. This
explains "rocBLAS works on gfx1013" reports (it is gfx1010 code) and why they fail on real
workloads, and is why the native-gfx1013 rocBLAS PR is worth landing.

## 4. Doorbell / SDMA knob

No `use_doorbell` module parameter or debugfs ring toggle exists on this stock kernel, and it is moot
regardless: SDMA is already out of the compute path (`HSA_ENABLE_SDMA=0`), and the wedge is on the
MEC compute queue, not SDMA. Consistent with the existing "no knob removed it".

## 5. Skipping the queue eviction removes the freeze (but does not fix compute)

An upstream amdgpu patch skips queue eviction on APUs to stop a gfx1103 evict/restore crash (rejected
upstream as a workaround). Applied here to gfx1013's nocpsch eviction path, it removes the board
**freeze**: across ~16 heavy sustained runs the board stayed alive and recoverable, where before it
would freeze and need a power-cycle. But it does not make compute reliable, completion degrades with
use (4/6 then 2/10 heavy runs completed), dispatches still hang, and silent wrong results persist.
Details and the patch: [`eviction-skip-experiment.md`](eviction-skip-experiment.md),
[`../../patches/kfd_skip_eviction_gfx1013.py`](../../patches/kfd_skip_eviction_gfx1013.py).

## 6. Memory placement and workarounds (follow-up)

`debug_evictions` confirms the wedge is a userptr eviction of GTT-mapped compute memory that the MEC
fails to preempt. Compute lands in GTT because the driver sets `apu_prefer_gtt` when VRAM < GTT (here
512 MB vs 16 GB). Forcing VRAM >= GTT (`bc250_memcfg UMA_SIZE 8192` for 8 GB VRAM **and**
`amdgpu.gttsize=4096` to cap GTT at 4 GB) flips that: compute then sits in VRAM (`VRAM_used` ~700 MB,
`GTT_used` ~12 MB during a GEMM), and the wedge drops from near-every heavy run to about 4 of 15. It is
the best mitigation found, and confirms the wedge is a GTT-eviction problem, but not a full fix (HIP's
auxiliary allocations stay in GTT), and it costs GTT the Vulkan path uses for large models. Raising the
carve-out alone (without capping GTT) does nothing. The ROCm#6386 gfx1103 workaround (`HSA_USE_SVM=0` +
GFXOFF-off + cwsr=0 + serialize) does not transfer to gfx1013 (no MES). Details:
[`memory-placement-and-workarounds.md`](memory-placement-and-workarounds.md).

## Net

No new fix, and no root cause established. The round adds a sharper picture of the fault itself (a
TCP/UTCL2 read permission fault on a mapped page), a minimal read-only reproducer, a
mainstream-framework demonstration (PyTorch sustained matmul wedges), a confirmed mechanism (a userptr
eviction of GTT compute memory that the no-MES gfx1013 MEC cannot preempt), and two partial
mitigations: skipping the KFD eviction (stops the board freeze), and forcing VRAM >= GTT so compute
sits in pinned VRAM (`bc250_memcfg` + `amdgpu.gttsize`, cuts the wedge to about 4 of 15). Neither is a
full fix. Vulkan remains the working path.

Historical, from the July investigation; collected by ad hoc invocations rather than a single harness, and kept for the record rather than as a reproducible run.
