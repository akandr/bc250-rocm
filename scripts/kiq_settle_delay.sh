#!/usr/bin/env bash
# Test whether the reset stall is a settle-time problem, by delaying the KIQ
# initialisation on the reset path.
#
# Result on this board: it is not. Fifty milliseconds does not help and five
# hundred does not either, so the device does not need wall-clock time. What did
# help, in an earlier cycle, was prints placed BETWEEN the register read and
# write inside gfx_v10_0_kiq_setting(), which suggests intervening bus activity
# rather than duration. The delay is left tunable at runtime via
# /sys/module/amdgpu/parameters/bc250_kiq_settle_ms so the question can be
# re-asked cheaply without a rebuild.
#
# Verify netconsole with a /dev/kmsg marker before triggering; restore the source
# and rebuild afterwards; check 40 CUs and a working benchmark.
set -eu
SRC=${SRC:-/home/akandr/k718/linux-7.1.8}
F=$SRC/drivers/gpu/drm/amd/amdgpu/gfx_v10_0.c
cp "$F" "$HOME/gfx_v10_0.c.presettle"
sudo python3 - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = """	gfx_v10_0_kiq_setting(ring);

	if (amdgpu_in_reset(adev)) { /* for GPU_RESET case */
		/* reset MQD to a clean status */
		if (adev->gfx.kiq[0].mqd_backup)"""
assert s.count(old) == 1
s = s.replace(old, """	if (amdgpu_in_reset(adev)) {
		dev_warn(adev->dev, "BC250SETTLE sleeping %u ms before KIQ init\\n",
			 bc250_kiq_settle_ms);
		msleep(bc250_kiq_settle_ms);
		dev_warn(adev->dev, "BC250SETTLE slept, proceeding\\n");
	}

	gfx_v10_0_kiq_setting(ring);
	dev_warn(adev->dev, "BC250SETTLE kiq_setting returned\\n");

	if (amdgpu_in_reset(adev)) { /* for GPU_RESET case */
		/* reset MQD to a clean status */
		if (adev->gfx.kiq[0].mqd_backup)""", 1)
i = s.rindex("#include"); j = s.index("\n", i)
s = s[:j+1] + """
static unsigned int bc250_kiq_settle_ms = 50;
module_param_named(bc250_kiq_settle_ms, bc250_kiq_settle_ms, uint, 0644);
MODULE_PARM_DESC(bc250_kiq_settle_ms,
	"BC-250: milliseconds to wait before KIQ init on the reset path (default 50)");
""" + s[j+1:]
old2 = """		nv_grbm_select(adev, 0, 0, 0, 0);
		mutex_unlock(&adev->srbm_mutex);
	} else {"""
assert s.count(old2) == 1
s = s.replace(old2, """		nv_grbm_select(adev, 0, 0, 0, 0);
		mutex_unlock(&adev->srbm_mutex);
		dev_warn(adev->dev, "BC250SETTLE reset path COMPLETE\\n");
	} else {""", 1)
open(p, "w").write(s); print("settle delay patch inserted")
PYEOF
cd "$SRC"
sudo make -C /lib/modules/$(uname -r)/build M="$SRC/drivers/gpu/drm/amd/amdgpu" -j6 modules 2>&1 | tail -3
