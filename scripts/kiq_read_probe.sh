#!/usr/bin/env bash
# Put the activity on both sides of the KIQ scheduler register read, and repeat
# every configuration.
#
# The previous build (kiq_gap_probe.sh) placed its knob between the read and the
# write, and its arms printed nothing at all: the read never returned, so the
# instrument was downstream of the stall. The prints that let an earlier run
# continue bracket the read rather than sitting between the accesses, so the one
# that could plausibly matter is the one BEFORE it. This build has a knob on each
# side, selectable at runtime:
#
#   bc250_kiq_pre_mode   before RREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS)
#   bc250_kiq_mid_mode   between that read and its matching write
#   0 nothing   1 msleep(gap_ms)   2 udelay busy-wait   3 dev_warn
#
# Repetition is the point, not placement. Every configuration tried so far has
# been run once, and single captures cannot tell a stopping point that depends on
# where the prints are from one that wanders between identical runs. Run each
# configuration several times before reading anything into the difference.
#
# Verify netconsole with a /dev/kmsg marker before triggering, trigger by READING
# /sys/kernel/debug/dri/*/amdgpu_gpu_recover (writing to it does nothing), restore
# the source and rebuild afterwards, and check 40 CUs and a working benchmark.
set -eu
SRC=${SRC:-/home/akandr/k718/linux-7.1.8}
F=$SRC/drivers/gpu/drm/amd/amdgpu/gfx_v10_0.c
cp "$HOME/gfx_v10_0.c.pregap" "$F"
sudo python3 - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()

helper = """
static unsigned int bc250_kiq_pre_mode;
static unsigned int bc250_kiq_mid_mode;
static unsigned int bc250_kiq_gap_ms = 50;
module_param_named(bc250_kiq_pre_mode, bc250_kiq_pre_mode, uint, 0644);
module_param_named(bc250_kiq_mid_mode, bc250_kiq_mid_mode, uint, 0644);
module_param_named(bc250_kiq_gap_ms, bc250_kiq_gap_ms, uint, 0644);
MODULE_PARM_DESC(bc250_kiq_pre_mode,
	"BC-250: what to do before the KIQ scheduler register read on the reset path: 0 nothing, 1 msleep, 2 udelay, 3 print");
MODULE_PARM_DESC(bc250_kiq_mid_mode,
	"BC-250: the same, between that read and its matching write");
MODULE_PARM_DESC(bc250_kiq_gap_ms,
	"BC-250: milliseconds for modes 1 and 2 (default 50)");

static void bc250_kiq_act(struct amdgpu_device *adev, unsigned int mode,
			  const char *where)
{
	unsigned int i;

	switch (mode) {
	case 1:
		msleep(bc250_kiq_gap_ms);
		break;
	case 2:
		for (i = 0; i < bc250_kiq_gap_ms; i++)
			udelay(1000);
		break;
	case 3:
		dev_warn(adev->dev, "BC250ACT %s\\n", where);
		break;
	default:
		break;
	}
}

static void gfx_v10_0_kiq_setting(struct amdgpu_ring *ring)"""
anchor = "\nstatic void gfx_v10_0_kiq_setting(struct amdgpu_ring *ring)"
assert s.count(anchor) == 1
s = s.replace(anchor, helper, 1)

old = """		tmp = RREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS);
		tmp &= 0xffffff00;"""
assert s.count(old) == 1
s = s.replace(old, """		if (amdgpu_in_reset(adev))
			bc250_kiq_act(adev, bc250_kiq_pre_mode, "before read");
		tmp = RREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS);
		if (amdgpu_in_reset(adev))
			bc250_kiq_act(adev, bc250_kiq_mid_mode, "after read");
		tmp &= 0xffffff00;""", 1)

steps = [
("""	if (amdgpu_in_reset(adev)) { /* for GPU_RESET case */
		/* reset MQD to a clean status */
		if (adev->gfx.kiq[0].mqd_backup)""",
 """	if (amdgpu_in_reset(adev)) { /* for GPU_RESET case */
		dev_warn(adev->dev, "BC250ACT step 1 kiq_setting returned\\n");
		/* reset MQD to a clean status */
		if (adev->gfx.kiq[0].mqd_backup)"""),
("""			memcpy_toio(mqd, adev->gfx.kiq[0].mqd_backup, sizeof(*mqd));

		/* reset ring buffer */""",
 """			memcpy_toio(mqd, adev->gfx.kiq[0].mqd_backup, sizeof(*mqd));

		dev_warn(adev->dev, "BC250ACT step 2 mqd restored\\n");
		/* reset ring buffer */"""),
("""		amdgpu_ring_clear_ring(ring);

		mutex_lock(&adev->srbm_mutex);
		nv_grbm_select(adev, ring->me, ring->pipe, ring->queue, 0);
		gfx_v10_0_kiq_init_register(ring);""",
 """		amdgpu_ring_clear_ring(ring);

		dev_warn(adev->dev, "BC250ACT step 3 ring cleared\\n");
		mutex_lock(&adev->srbm_mutex);
		nv_grbm_select(adev, ring->me, ring->pipe, ring->queue, 0);
		dev_warn(adev->dev, "BC250ACT step 4 grbm selected\\n");
		gfx_v10_0_kiq_init_register(ring);
		dev_warn(adev->dev, "BC250ACT step 5 init_register returned\\n");"""),
("""		nv_grbm_select(adev, 0, 0, 0, 0);
		mutex_unlock(&adev->srbm_mutex);
	} else {""",
 """		nv_grbm_select(adev, 0, 0, 0, 0);
		mutex_unlock(&adev->srbm_mutex);
		dev_warn(adev->dev, "BC250ACT step 6 reset path COMPLETE\\n");
	} else {"""),
]
for a, b in steps:
    assert s.count(a) == 1, a[:40]
    s = s.replace(a, b, 1)

open(p, "w").write(s); print("read probe patch inserted")
PYEOF
sudo make -C /lib/modules/$(uname -r)/build M="$SRC/drivers/gpu/drm/amd/amdgpu" -j6 modules 2>&1 | tail -2
