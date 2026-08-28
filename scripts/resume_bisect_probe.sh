#!/usr/bin/env bash
# Locate a host stall during GPU reset recovery by naming each IP block as it
# resumes.
#
# The stall leaves no clue in the log: every capture ends at the same place and
# dynamic debug reaches only into the SMU resume, because the path beyond has no
# pr_debug to enable. Two prints around the resume loop are enough to bisect it,
# since a block that is entered and never returns identifies itself by the
# missing line.
#
# Build with this, install with scripts/install_probe_module.sh, reboot, arm
# scripts/netconsole_capture.sh, then trigger a reset through
# /sys/kernel/debug/dri/0/amdgpu_gpu_recover. The probe fires only during a
# reset, so it is inert in normal operation.
set -eu
SRC=${SRC:-/home/akandr/k718/linux-7.1.8}
F=$SRC/drivers/gpu/drm/amd/amdgpu/amdgpu_device.c
cp "$F" "$HOME/amdgpu_device.c.prebisect"
sudo python3 - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
anchor = """		r = amdgpu_ip_block_resume(&adev->ip_blocks[i]);
		if (r)
			return r;
	}

	return 0;
}

/**
 * amdgpu_device_ip_resume_phase3 - run resume for hardware IPs"""
assert s.count(anchor) == 1, s.count(anchor)
add = """		dev_warn(adev->dev, "BC250RESUME phase2 entering block %d <%s>\\n",
			 i, adev->ip_blocks[i].version->funcs->name);
		r = amdgpu_ip_block_resume(&adev->ip_blocks[i]);
		dev_warn(adev->dev, "BC250RESUME phase2 block %d <%s> returned %d\\n",
			 i, adev->ip_blocks[i].version->funcs->name, r);
		if (r)
			return r;
	}

	dev_warn(adev->dev, "BC250RESUME phase2 complete\\n");
	return 0;
}

/**
 * amdgpu_device_ip_resume_phase3 - run resume for hardware IPs"""
open(p, "w").write(s.replace(anchor, add, 1))
print("resume bisect probe inserted")
PYEOF
cd "$SRC"
sudo make -C /lib/modules/$(uname -r)/build M="$SRC/drivers/gpu/drm/amd/amdgpu" -j6 modules 2>&1 | tail -3
