#!/usr/bin/env bash
# Bisect inside the GFX block's resume, one level below the IP-block bisect.
#
# scripts/resume_bisect_probe.sh showed that gfx_v10_0's resume callback is
# entered and never returns. This adds prints around the three main steps of
# gfx_v10_0_hw_init, which the resume callback simply forwards to:
# constants_init, rlc_resume and cp_resume. The step that is entered without a
# matching return line is where the host stalls.
#
# Build with this, install with scripts/install_probe_module.sh, reboot, then
# arm scripts/netconsole_capture.sh and VERIFY IT by writing a marker to
# /dev/kmsg and confirming it reaches the listener. Arming netconsole while the
# board is still booting produces an empty capture that reads like a result.
# Restore the source and rebuild afterwards.
set -eu
SRC=${SRC:-/home/akandr/k718/linux-7.1.8}
F=$SRC/drivers/gpu/drm/amd/amdgpu/gfx_v10_0.c
cp "$F" "$HOME/gfx_v10_0.c.prebisect"
sudo python3 - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = """	gfx_v10_0_constants_init(adev);

	r = gfx_v10_0_rlc_resume(adev);
	if (r)
		return r;
"""
assert s.count(old) == 1
s = s.replace(old, """	dev_warn(adev->dev, "BC250GFX before constants_init\\n");
	gfx_v10_0_constants_init(adev);
	dev_warn(adev->dev, "BC250GFX after constants_init, before rlc_resume\\n");

	r = gfx_v10_0_rlc_resume(adev);
	dev_warn(adev->dev, "BC250GFX rlc_resume returned %d\\n", r);
	if (r)
		return r;
""", 1)
old2 = """	r = gfx_v10_0_cp_resume(adev);
	if (r)
		return r;
"""
assert s.count(old2) == 1
s = s.replace(old2, """	dev_warn(adev->dev, "BC250GFX before cp_resume\\n");
	r = gfx_v10_0_cp_resume(adev);
	dev_warn(adev->dev, "BC250GFX cp_resume returned %d\\n", r);
	if (r)
		return r;
""", 1)
open(p, "w").write(s); print("gfx resume probes inserted")
PYEOF
cd "$SRC"
sudo make -C /lib/modules/$(uname -r)/build M="$SRC/drivers/gpu/drm/amd/amdgpu" -j6 modules 2>&1 | tail -3
