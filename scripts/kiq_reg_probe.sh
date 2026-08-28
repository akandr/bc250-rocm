#!/usr/bin/env bash
# Is any GPU register readable at the point where the KIQ resume hangs?
#
# After a reset the host stalls inside a single MMIO read, RREG32_SOC15(GC, 0,
# mmRLC_CP_SCHEDULERS) in gfx_v10_0_kiq_setting(): a print immediately before it
# lands and nothing after it ever does. That does not say whether the register
# block is unreachable or whether this one register is. The difference matters,
# because an unreachable block points at power or clock state left wrong by the
# reset, while one register points at the RLC.
#
# This build reads three registers in increasing order of specificity just before
# the scheduler read, printing each value as it returns, so the first one that
# hangs names the boundary:
#
#   mmGRBM_STATUS     global, outside the RLC
#   mmSCRATCH_REG0    a scratch register, harmless to read
#   mmRLC_STAT        the RLC block itself, where the hanging register lives
#
# Every line carries a sequence number, because netconsole is UDP and a dropped
# packet is indistinguishable from a stall: one capture in this campaign lost its
# final watchdog line, and the reasoning "this printed and nothing after it did"
# rests entirely on delivery. A gap in the numbers says packet loss, an ordered
# run that simply stops says stall.
#
# Enabled at runtime with bc250_kiq_regprobe=1 so the same module can run the
# control. Repeat every configuration: the observation this replaces did not
# reproduce, which is why the reading built on it was wrong.
#
# Verify netconsole with a /dev/kmsg marker before triggering, trigger by READING
# /sys/kernel/debug/dri/*/amdgpu_gpu_recover, restore the source and rebuild
# afterwards, and check 40 CUs and a working benchmark.
set -eu
SRC=${SRC:-/home/akandr/k718/linux-7.1.8}
F=$SRC/drivers/gpu/drm/amd/amdgpu/gfx_v10_0.c
for r in mmGRBM_STATUS mmSCRATCH_REG0 mmRLC_STAT; do
	grep -rq "define $r\b" "$SRC/drivers/gpu/drm/amd/include/asic_reg/gc/" || {
		echo "register macro $r not found for this ASIC, refusing to patch"; exit 1; }
done
cp "$HOME/gfx_v10_0.c.pregap" "$F"
sudo python3 - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()

helper = """
static unsigned int bc250_kiq_regprobe;
module_param_named(bc250_kiq_regprobe, bc250_kiq_regprobe, uint, 0644);
MODULE_PARM_DESC(bc250_kiq_regprobe,
	"BC-250: read three registers before the KIQ scheduler read on the reset path, printing each");

static void bc250_kiq_regs(struct amdgpu_device *adev)
{
	uint32_t v, cfg;

	if (!bc250_kiq_regprobe)
		return;

	/* Before touching the GPU at all, ask the bus whether it is still there.
	 * A config-space read does not go through the BAR, so it answers a
	 * different question than any MMIO read can: 0x1002 means the device is
	 * present and the link is up, and all-ones means it has fallen off the
	 * bus and every later hang is a consequence rather than a cause.
	 */
	pci_read_config_dword(adev->pdev, 0, &cfg);
	dev_warn(adev->dev, "BC250REG %u pci config dword0 = 0x%08x\\n", 1, cfg);
	pci_read_config_dword(adev->pdev, PCI_COMMAND, &cfg);
	dev_warn(adev->dev, "BC250REG %u pci command/status = 0x%08x\\n", 2, cfg);

	/* Whether the access goes through the MMIO window or the PCIE index/data
	 * pair decides how fragile it is after a reset, and that depends only on
	 * where the offset falls relative to the mapped window.
	 */
	dev_warn(adev->dev, "BC250REG %u rmmio_size=%u sched_off=0x%x grbm_off=0x%x\\n", 3,
		 (unsigned int)adev->rmmio_size,
		 SOC15_REG_OFFSET(GC, 0, mmRLC_CP_SCHEDULERS),
		 SOC15_REG_OFFSET(GC, 0, mmGRBM_STATUS));

	dev_warn(adev->dev, "BC250REG %u about to read GRBM_STATUS\\n", 4);
	v = RREG32_SOC15(GC, 0, mmGRBM_STATUS);
	dev_warn(adev->dev, "BC250REG %u GRBM_STATUS = 0x%08x\\n", 5, v);

	dev_warn(adev->dev, "BC250REG %u about to read SCRATCH_REG0\\n", 6);
	v = RREG32_SOC15(GC, 0, mmSCRATCH_REG0);
	dev_warn(adev->dev, "BC250REG %u SCRATCH_REG0 = 0x%08x\\n", 7, v);

	dev_warn(adev->dev, "BC250REG %u about to read RLC_STAT\\n", 8);
	v = RREG32_SOC15(GC, 0, mmRLC_STAT);
	dev_warn(adev->dev, "BC250REG %u RLC_STAT = 0x%08x\\n", 9, v);
}

static void gfx_v10_0_kiq_setting(struct amdgpu_ring *ring)"""
anchor = "\nstatic void gfx_v10_0_kiq_setting(struct amdgpu_ring *ring)"
assert s.count(anchor) == 1
s = s.replace(anchor, helper, 1)

old = """		tmp = RREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS);
		tmp &= 0xffffff00;"""
assert s.count(old) == 1
s = s.replace(old, """		if (amdgpu_in_reset(adev)) {
			bc250_kiq_regs(adev);
			dev_warn(adev->dev, "BC250REG %u about to read RLC_CP_SCHEDULERS\\n", 10);
		}
		tmp = RREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS);
		if (amdgpu_in_reset(adev))
			dev_warn(adev->dev, "BC250REG %u RLC_CP_SCHEDULERS = 0x%08x\\n", 11, tmp);
		tmp &= 0xffffff00;""", 1)

open(p, "w").write(s); print("register probe patch inserted")
PYEOF
sudo make -C /lib/modules/$(uname -r)/build M="$SRC/drivers/gpu/drm/amd/amdgpu" -j6 modules 2>&1 | tail -2
