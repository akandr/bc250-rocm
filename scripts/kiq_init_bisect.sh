#!/usr/bin/env bash
# Bisect inside gfx_v10_0_kiq_init_queue(), where the reset stall lives.
#
# gfx_v10_0_kiq_resume() is a one-line wrapper, so the seven steps of the reset
# path in kiq_init_queue are the useful granularity, plus a split of the register
# read and write inside kiq_setting.
#
# IMPORTANT: the failure point moves with timing. Adding these prints, each of
# which netconsole ships over UDP, lets the sequence get six steps further than
# it does with fewer prints. Do not read the last line as "the broken
# instruction"; read it as how far this build got. That behaviour is itself the
# finding, and suggests trying a delay before KIQ init as a fix.
#
# Verify netconsole with a /dev/kmsg marker before triggering; restore the source
# and rebuild afterwards; check 40 CUs and a working benchmark.
set -eu
SRC=${SRC:-/home/akandr/k718/linux-7.1.8}
F=$SRC/drivers/gpu/drm/amd/amdgpu/gfx_v10_0.c
cp "$F" "$HOME/gfx_v10_0.c.prekiqbisect"
sudo python3 - "$F" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
old_reg = """	default:
		tmp = RREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS);
		tmp &= 0xffffff00;
		tmp |= (ring->me << 5) | (ring->pipe << 3) | (ring->queue);
		WREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS, tmp | 0x80);
		break;
	}
}"""
assert s.count(old_reg) == 1
s = s.replace(old_reg, """	default:
		dev_warn(adev->dev, "BC250KIQ 1a before RREG32 mmRLC_CP_SCHEDULERS\\n");
		tmp = RREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS);
		dev_warn(adev->dev, "BC250KIQ 1b read returned 0x%08x\\n", tmp);
		tmp &= 0xffffff00;
		tmp |= (ring->me << 5) | (ring->pipe << 3) | (ring->queue);
		WREG32_SOC15(GC, 0, mmRLC_CP_SCHEDULERS, tmp | 0x80);
		dev_warn(adev->dev, "BC250KIQ 1c write complete\\n");
		break;
	}
}""", 1)
old_seq = """	gfx_v10_0_kiq_setting(ring);

	if (amdgpu_in_reset(adev)) { /* for GPU_RESET case */
		/* reset MQD to a clean status */
		if (adev->gfx.kiq[0].mqd_backup)
			memcpy_toio(mqd, adev->gfx.kiq[0].mqd_backup, sizeof(*mqd));

		/* reset ring buffer */
		ring->wptr = 0;
		amdgpu_ring_clear_ring(ring);

		mutex_lock(&adev->srbm_mutex);
		nv_grbm_select(adev, ring->me, ring->pipe, ring->queue, 0);
		gfx_v10_0_kiq_init_register(ring);
		nv_grbm_select(adev, 0, 0, 0, 0);
		mutex_unlock(&adev->srbm_mutex);
	} else {"""
assert s.count(old_seq) == 1
s = s.replace(old_seq, """	dev_warn(adev->dev, "BC250KIQ 1 before kiq_setting\\n");
	gfx_v10_0_kiq_setting(ring);
	dev_warn(adev->dev, "BC250KIQ 2 after kiq_setting, in_reset=%d\\n",
		 amdgpu_in_reset(adev) ? 1 : 0);

	if (amdgpu_in_reset(adev)) { /* for GPU_RESET case */
		dev_warn(adev->dev, "BC250KIQ 3 before memcpy_toio backup=%p\\n",
			 adev->gfx.kiq[0].mqd_backup);
		if (adev->gfx.kiq[0].mqd_backup)
			memcpy_toio(mqd, adev->gfx.kiq[0].mqd_backup, sizeof(*mqd));
		dev_warn(adev->dev, "BC250KIQ 4 after memcpy_toio\\n");

		ring->wptr = 0;
		amdgpu_ring_clear_ring(ring);
		dev_warn(adev->dev, "BC250KIQ 5 after clear_ring\\n");

		mutex_lock(&adev->srbm_mutex);
		dev_warn(adev->dev, "BC250KIQ 6 got srbm_mutex\\n");
		nv_grbm_select(adev, ring->me, ring->pipe, ring->queue, 0);
		dev_warn(adev->dev, "BC250KIQ 7 after grbm_select\\n");
		gfx_v10_0_kiq_init_register(ring);
		dev_warn(adev->dev, "BC250KIQ 8 after kiq_init_register\\n");
		nv_grbm_select(adev, 0, 0, 0, 0);
		mutex_unlock(&adev->srbm_mutex);
		dev_warn(adev->dev, "BC250KIQ 9 reset path complete\\n");
	} else {""", 1)
open(p, "w").write(s); print("kiq init probes inserted")
PYEOF
cd "$SRC"
sudo make -C /lib/modules/$(uname -r)/build M="$SRC/drivers/gpu/drm/amd/amdgpu" -j6 modules 2>&1 | tail -3
