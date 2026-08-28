#!/usr/bin/env bash
# Separate "time at that point" from "traffic at that point" in the KIQ resume.
#
# A delay placed before the KIQ initialisation does nothing (kiq_settle_delay.sh),
# so the board does not want wall-clock time on that path. What did let the
# sequence run further was prints placed BETWEEN the register read and the
# matching write inside gfx_v10_0_kiq_setting(). A dev_warn there means printk
# work and a netconsole UDP packet interleaved between two GPU register accesses,
# which is delay and bus activity together. This build varies only what sits
# between those two accesses, at runtime, so the two can be told apart:
#
#   bc250_kiq_gap_mode=0  nothing            control, should stall in kiq_setting
#   bc250_kiq_gap_mode=1  msleep(gap_ms)     delay with scheduling, no bus traffic
#   bc250_kiq_gap_mode=2  udelay busy-wait   delay without scheduling
#   bc250_kiq_gap_mode=3  dev_warn           positive control, the known-helpful case
#
# Mode 3 matters as much as the others. Nothing here has ever made the board
# survive a reset; "helped" only ever meant "got further", so a null result from
# modes 1 and 2 is uninterpretable unless mode 3 still gets further in this same
# build. The step markers in kiq_init_queue are the ruler that measures how far.
#
# Verify netconsole with a /dev/kmsg marker before triggering; restore the source
# and rebuild afterwards; check 40 CUs and a working benchmark.
set -eu
SRC=${SRC:-/home/akandr/k718/linux-7.1.8}
F=$SRC/drivers/gpu/drm/amd/amdgpu/gfx_v10_0.c
cp "$F" "$HOME/gfx_v10_0.c.pregap"
sudo python3 - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()

helper = """
static unsigned int bc250_kiq_gap_mode;
static unsigned int bc250_kiq_gap_ms = 50;
module_param_named(bc250_kiq_gap_mode, bc250_kiq_gap_mode, uint, 0644);
module_param_named(bc250_kiq_gap_ms, bc250_kiq_gap_ms, uint, 0644);
MODULE_PARM_DESC(bc250_kiq_gap_mode,
	"BC-250: what to put between the KIQ scheduler register read and write on the reset path: 0 nothing, 1 msleep, 2 udelay, 3 print");
MODULE_PARM_DESC(bc250_kiq_gap_ms,
	"BC-250: milliseconds for gap modes 1 and 2 (default 50)");

static void bc250_kiq_gap(struct amdgpu_device *adev)
{
	unsigned int i;

	switch (bc250_kiq_gap_mode) {
	case 1:
		msleep(bc250_kiq_gap_ms);
		break;
	case 2:
		for (i = 0; i < bc250_kiq_gap_ms; i++)
			udelay(1000);
		break;
	case 3:
		dev_warn(adev->dev, "BC250GAP between read and write\\n");
		break;
	default:
		break;
	}
}

static void gfx_v10_0_kiq_setting(struct amdgpu_ring *ring)"""
anchor = "\nstatic void gfx_v10_0_kiq_setting(struct amdgpu_ring *ring)"
assert s.count(anchor) == 1, s.count(anchor)
s = s.replace(anchor, helper, 1)

old = """		tmp = RREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS);
		tmp &= 0xffffff00;"""
assert s.count(old) == 1
s = s.replace(old, """		tmp = RREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS);
		if (amdgpu_in_reset(adev))
			bc250_kiq_gap(adev);
		tmp &= 0xffffff00;""", 1)

steps = [
("""	if (amdgpu_in_reset(adev)) { /* for GPU_RESET case */
		/* reset MQD to a clean status */
		if (adev->gfx.kiq[0].mqd_backup)""",
 """	if (amdgpu_in_reset(adev)) { /* for GPU_RESET case */
		dev_warn(adev->dev, "BC250GAP step 1 kiq_setting returned\\n");
		/* reset MQD to a clean status */
		if (adev->gfx.kiq[0].mqd_backup)"""),
("""			memcpy_toio(mqd, adev->gfx.kiq[0].mqd_backup, sizeof(*mqd));

		/* reset ring buffer */""",
 """			memcpy_toio(mqd, adev->gfx.kiq[0].mqd_backup, sizeof(*mqd));

		dev_warn(adev->dev, "BC250GAP step 2 mqd restored\\n");
		/* reset ring buffer */"""),
("""		amdgpu_ring_clear_ring(ring);

		mutex_lock(&adev->srbm_mutex);
		nv_grbm_select(adev, ring->me, ring->pipe, ring->queue, 0);
		gfx_v10_0_kiq_init_register(ring);""",
 """		amdgpu_ring_clear_ring(ring);

		dev_warn(adev->dev, "BC250GAP step 3 ring cleared\\n");
		mutex_lock(&adev->srbm_mutex);
		nv_grbm_select(adev, ring->me, ring->pipe, ring->queue, 0);
		dev_warn(adev->dev, "BC250GAP step 4 grbm selected\\n");
		gfx_v10_0_kiq_init_register(ring);
		dev_warn(adev->dev, "BC250GAP step 5 init_register returned\\n");"""),
("""		nv_grbm_select(adev, 0, 0, 0, 0);
		mutex_unlock(&adev->srbm_mutex);
	} else {""",
 """		nv_grbm_select(adev, 0, 0, 0, 0);
		mutex_unlock(&adev->srbm_mutex);
		dev_warn(adev->dev, "BC250GAP step 6 reset path COMPLETE\\n");
	} else {"""),
]
for a, b in steps:
    assert s.count(a) == 1, a[:40]
    s = s.replace(a, b, 1)

open(p, "w").write(s); print("gap probe patch inserted")
PYEOF
sudo make -C /lib/modules/$(uname -r)/build M="$SRC/drivers/gpu/drm/amd/amdgpu" -j6 modules 2>&1 | tail -3
