#!/usr/bin/env bash
# Bisect inside gfx_v10_0_cp_resume(), the last level of the reset-stall hunt.
#
# scripts/gfx_resume_bisect.sh showed that cp_resume is entered and never
# returns. Its sequence is microcode load, KIQ resume, KCQ resume, then the
# graphics ring path, so prints around the three resume calls name the step that
# does not come back. On this board it is gfx_v10_0_kiq_resume().
#
# Build with this, install with scripts/install_probe_module.sh, reboot, arm
# scripts/netconsole_capture.sh and VERIFY it with a marker written to
# /dev/kmsg before triggering. An unverified empty capture reads exactly like a
# result. Restore the source and rebuild afterwards, then check 40 CUs and a
# working benchmark.
set -eu
SRC=${SRC:-/home/akandr/k718/linux-7.1.8}
F=$SRC/drivers/gpu/drm/amd/amdgpu/gfx_v10_0.c
cp "$F" "$HOME/gfx_v10_0.c.precpbisect"
sudo python3 - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
subs = [
 ("""	r = gfx_v10_0_kiq_resume(adev);
	if (r)
		return r;
""",
  """	dev_warn(adev->dev, "BC250CP before kiq_resume\\n");
	r = gfx_v10_0_kiq_resume(adev);
	dev_warn(adev->dev, "BC250CP kiq_resume returned %d\\n", r);
	if (r)
		return r;
"""),
 ("""	r = gfx_v10_0_kcq_resume(adev);
	if (r)
		return r;
""",
  """	dev_warn(adev->dev, "BC250CP before kcq_resume\\n");
	r = gfx_v10_0_kcq_resume(adev);
	dev_warn(adev->dev, "BC250CP kcq_resume returned %d\\n", r);
	if (r)
		return r;
"""),
 ("""		r = gfx_v10_0_cp_gfx_resume(adev);
		if (r)
			return r;
""",
  """		dev_warn(adev->dev, "BC250CP before cp_gfx_resume\\n");
		r = gfx_v10_0_cp_gfx_resume(adev);
		dev_warn(adev->dev, "BC250CP cp_gfx_resume returned %d\\n", r);
		if (r)
			return r;
"""),
]
for old, new in subs:
    assert s.count(old) == 1, (s.count(old), old[:40])
    s = s.replace(old, new, 1)
open(p, "w").write(s); print("cp resume probes inserted")
PYEOF
cd "$SRC"
sudo make -C /lib/modules/$(uname -r)/build M="$SRC/drivers/gpu/drm/amd/amdgpu" -j6 modules 2>&1 | tail -3
