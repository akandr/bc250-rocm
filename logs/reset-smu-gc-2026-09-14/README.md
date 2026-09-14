# The reset that is not a reset, and two attempts at a real one, 2026-09-14

Captured over netconsole with [`scripts/reset_trial_repeat.sh`](../../scripts/reset_trial_repeat.sh),
kernel 7.1.8, the working configuration of the front page (`bc250_flush_by_runlist=3`,
`bc250_flush_pasid_kiq=0`, `bc250_cc_write_mode=3`, `gpu_recovery=0`, navi12 SDMA microcode),
GPU idle, reset requested by reading `amdgpu_gpu_recover` in debugfs (`Source: 5`).

## What the source says, before any capture

Neither reset method amdgpu offers for this chip reaches the hardware.

- **MODE1** calls `amdgpu_device_mode1_reset()`, which calls `psp_gpu_reset()`, which calls
  `psp_mode1_reset()`. That macro is
  `((psp)->funcs->mode1_reset ? (psp)->funcs->mode1_reset((psp)) : false)`, and the PSP
  implementation Cyan Skillfish2 is given, `psp_v11_0_8_funcs`, defines only the five ring
  callbacks. The macro evaluates to `false`, which is 0, which the caller treats as success.
- **MODE2** calls `nv_asic_mode2_reset()`, which calls `smu_mode2_reset()`, which calls
  `ppt_funcs->mode2_reset` only if it exists. `cyan_skillfish_ppt_funcs` has none, so `ret` stays 0.

Both then log `GPU reset succeeded, trying to resume`, and the driver re-initialises a GPU that
was never reset. The earlier captures already carried the evidence without it being read that
way. MODE1 goes from `GPU psp mode1 reset` to `GPU reset succeeded` in 95 microseconds
([`../reset-path-2026-08-24/path-kfd_rep1.log`](../reset-path-2026-08-24/path-kfd_rep1.log)), MODE2
in 160 ([`../reset-netconsole-2026-08-23/mode2.log`](../reset-netconsole-2026-08-23/mode2.log)).
And the 40-CU unlock prints a register it rewrites, `SPI`, which reads `0x00000007` at boot and is
written to `0x0000001f`. After a real reset it would read `0x00000007` again. In every
post-reset capture it reads `0x0000001f`: the value written at boot survived the "reset".

A second upstream detail matters for anyone running the default. `amdgpu_device_should_recover_gpu()`
lists `CHIP_CYAN_SKILLFISH` among chips whose recovery is disabled when `gpu_recovery=-1`, but an
earlier line, `if (!amdgpu_ras_is_poison_mode_supported(adev)) return true;`, returns before that
list is reached on any device without a RAS context, which includes this APU. The line arrived with
commit 1a11a65d5395 ("Enable mode-1 reset for RAS recovery in fatal error mode", first in v6.2) and
is unchanged on master as of this date. That is why the KFD probe of 24 August reported
`gpu_recovery=-1, should_recover=1`. Whether the list or the early return reflects the intended
behaviour is for upstream to say; as written, the list is unreachable for this chip.

## Arm 1: SMU message 0x2E as the MODE2 reset

The PMFW header for this chip, `smu_v11_8_ppsmc.h`, defines `PPSMC_MSG_InitiateGcRsmuSoftReset`
(0x2E), which no driver code uses.
[`scripts/apply_smu_gc_reset.py`](../../scripts/apply_smu_gc_reset.py) maps it and adds a
`mode2_reset` that sends it behind a runtime parameter.

- `smu2e-sync_rep1.log` is **not** an arm of this experiment. The patched module had been installed
  but the board had not been rebooted into it, so the parameter write failed silently and the
  trial ran the production module. It is kept as a same-day control: the no-op MODE2, `SPI`
  reading `0x1f`, and the usual watchdog stall.
- `smu2e-sync_rep2.log` ran the patched module. The SMU firmware (88.6.0) rejects the message:
  `SMU: unknown command msg_reg: 2e resp_reg: fe`, `ret=-95`. Nothing was reset, `SPI` reads
  `0x1f`, and the host stalls as before. The message exists in the header and not in this
  firmware.

## Arm 2: PCI function-level reset, `reset_method=6`

The function advertises `flr bus` in `reset_method`, and `nv_asic_reset_method()` accepts
`AMD_RESET_METHOD_PCI`, which calls `pci_reset_function()`.

- `pci-flr_rep1.log`: the FLR is issued and the function never comes back. `not ready ... after
  FLR` doubles up to 65535 ms, then `ASIC reset failed with error, -25`. **The host survived**,
  which no earlier reset attempt on this board did.
- `pci-flr_rep2.log`: same boot, second trigger, `device lost from bus!`, `ret = -19`, host still
  up. This is the aftermath of rep1 rather than a second FLR.

Afterwards, on the same boot and outside the harness (no capture file): config space of 01:00.0
read all `0xff`; writing `bus` to its `reset_method` was refused; a manual secondary bus reset,
pulsing bit 6 of `BRIDGE_CONTROL` on the bridge 00:08.1, left both 01:00.0 and 01:00.1 reading
`0xff` for the following 30 seconds. Only a power cycle brought the GPU back.

## What this changes

The reset defect is not a stall in bringing up a freshly reset GPU. On this chip the driver has
no reset to perform, reports success anyway, and then reinitialises live hardware, which is where
the first GC register access stops returning. That also explains the intermittency and the
failure of every settle delay: nothing was settling.

Of the two primitives tried, one does not exist in the firmware and the other removes the
function from the bus until power-off while keeping the host alive. `amdgpu.gpu_recovery=0`
remains the practical setting. Two trials per arm, one board.
