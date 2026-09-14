#!/usr/bin/env python3
# Give Cyan Skillfish a MODE2 reset that does something.
#
# On this chip neither reset amdgpu offers touches the hardware. MODE1 goes
# through psp_gpu_reset(), and psp_v11_0_8 has no .mode1_reset, so the
# psp_mode1_reset() macro evaluates to false, which the caller reads as success.
# MODE2 goes through smu_mode2_reset(), and cyan_skillfish_ppt has no
# .mode2_reset, so ret stays 0. Both log "GPU reset succeeded" within 0.1 ms and
# the driver then re-initialises a GPU that was never reset. The captures agree:
# the SPI register the 40-CU unlock writes from 0x7 to 0x1f at boot still reads
# 0x1f when the unlock runs again after the "reset".
#
# The PMFW header for this chip defines PPSMC_MSG_InitiateGcRsmuSoftReset (0x2E),
# which the driver never maps. This adds it as the MODE2 reset, behind a runtime
# parameter so the stock no-op remains the control arm on the same module:
#
#   amdgpu.bc250_smu_reset=0   stock: mode2_reset does nothing (default)
#   amdgpu.bc250_smu_reset=1   send 0x2E and wait for the SMU response
#   amdgpu.bc250_smu_reset=2   send 0x2E without waiting (the reset may take the
#                              SMU response path down with it)
#   amdgpu.bc250_smu_reset_arg     message parameter, default 0
#   amdgpu.bc250_smu_reset_delay   milliseconds to sleep after sending, default 100
#
# Select MODE2 at runtime with: echo 3 > /sys/module/amdgpu/parameters/reset_method
# Whether a reset happened is readable from the existing unlock print: SPI reading
# 0x00000007 before the rewrite means the register was reset, 0x0000001f means not.
#
# Usage: apply_smu_gc_reset.py <tree>/drivers/gpu/drm/amd   (idempotent)
import sys, pathlib

amd = pathlib.Path(sys.argv[1])
f = amd / "pm/swsmu/smu11/cyan_skillfish_ppt.c"
s = f.read_text()
if "bc250_smu_reset" in s:
    print("already patched"); sys.exit(0)

anchor = "\tMSG_MAP(UnforceGfxVid,                  PPSMC_MSG_UnforceGfxVid,\t\t0),\n"
assert anchor in s, "message map anchor not found"
s = s.replace(anchor, anchor +
    "\tMSG_MAP(GfxDeviceDriverReset,           PPSMC_MSG_InitiateGcRsmuSoftReset,\t0),\n")

func = r'''
/* BC-250: see scripts/apply_smu_gc_reset.py in akandr/bc250-rocm */
static int bc250_smu_reset;
module_param(bc250_smu_reset, int, 0644);
MODULE_PARM_DESC(bc250_smu_reset, "BC-250 MODE2 reset: 0 stock no-op, 1 SMU 0x2E sync, 2 SMU 0x2E async");
static int bc250_smu_reset_arg;
module_param(bc250_smu_reset_arg, int, 0644);
MODULE_PARM_DESC(bc250_smu_reset_arg, "BC-250 parameter for SMU message 0x2E");
static int bc250_smu_reset_delay = 100;
module_param(bc250_smu_reset_delay, int, 0644);
MODULE_PARM_DESC(bc250_smu_reset_delay, "BC-250 ms to sleep after SMU message 0x2E");

static int cyan_skillfish_mode2_reset(struct smu_context *smu)
{
	struct smu_msg_ctl *ctl = &smu->msg_ctl;
	u32 out = 0;
	int ret = 0;

	dev_info(smu->adev->dev, "BC250SMURESET mode=%d arg=%d delay=%d\n",
		 bc250_smu_reset, bc250_smu_reset_arg, bc250_smu_reset_delay);
	if (bc250_smu_reset == 1) {
		ret = smu_cmn_send_smc_msg_with_param(smu, SMU_MSG_GfxDeviceDriverReset,
						      bc250_smu_reset_arg, &out);
	} else if (bc250_smu_reset == 2) {
		mutex_lock(&ctl->lock);
		ret = smu_msg_send_async_locked(ctl, SMU_MSG_GfxDeviceDriverReset,
						bc250_smu_reset_arg);
		mutex_unlock(&ctl->lock);
	} else {
		return 0;
	}
	dev_info(smu->adev->dev, "BC250SMURESET sent ret=%d out=0x%08x\n", ret, out);
	if (bc250_smu_reset_delay > 0)
		msleep(bc250_smu_reset_delay);
	dev_info(smu->adev->dev, "BC250SMURESET slept, returning 0\n");
	/* return 0 either way so the resume runs and shows what the message did */
	return 0;
}
'''
anchor2 = "static const struct pptable_funcs cyan_skillfish_ppt_funcs = {\n"
assert anchor2 in s
s = s.replace(anchor2, func + "\n" + anchor2 + "\t.mode2_reset = cyan_skillfish_mode2_reset,\n")
if "#include <linux/delay.h>" not in s:
    s = s.replace('#include "amdgpu.h"', '#include <linux/delay.h>\n#include "amdgpu.h"', 1)
f.write_text(s)
print("patched", f)
