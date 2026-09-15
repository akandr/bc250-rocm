#!/usr/bin/env python3
# Port gfx9's per-queue compute reset to gfx10, behind a runtime parameter.
#
# On gfx10, kgd_gfx_v10_hqd_get_pq_addr() and kgd_gfx_v10_hqd_reset() are stubs that
# return 0, so when a compute queue refuses preemption under hardware scheduling,
# reset_hung_queues() finds nothing it can reset and KFD escalates to a full GPU
# reset. On Cyan Skillfish there is no working full reset (see
# logs/reset-smu-gc-2026-09-14/), so a single hung queue ends either in a hung host
# (gpu_recovery at the default) or a GPU that stays unusable until reboot
# (gpu_recovery=0).
#
# gfx9 implements both (commit ee0a469cf917, "support per-queue reset on gfx9"):
# write SPI_COMPUTE_QUEUE_RESET, wait for CP_HQD_ACTIVE to clear, and if it does not,
# reset the pipe through CP_MEC_CNTL. The same registers exist in gc_10_1_0. This is
# that code with the gfx10 queue-select helpers.
#
#   amdgpu.bc250_queue_reset=0   stubs, stock behaviour (default)
#   amdgpu.bc250_queue_reset=1   gfx9-style per-queue reset
#
# Note reset_queues_on_hws_hang() returns -ENOTRECOVERABLE when gpu_recovery=0, so
# this is only reachable with gpu_recovery at the default or 1.
#
# Usage: apply_gfx10_queue_reset.py <tree>/drivers/gpu/drm/amd   (idempotent)
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "amdgpu/amdgpu_amdkfd_gfx_v10.c"
s = p.read_text()
if "bc250_queue_reset" in s:
    print("already patched"); sys.exit(0)

old = '''uint64_t kgd_gfx_v10_hqd_get_pq_addr(struct amdgpu_device *adev,
				     uint32_t pipe_id, uint32_t queue_id,
				     uint32_t inst)
{
	return 0;
}

uint64_t kgd_gfx_v10_hqd_reset(struct amdgpu_device *adev,
			       uint32_t pipe_id, uint32_t queue_id,
			       uint32_t inst, unsigned int utimeout)
{
	return 0;
}
'''
assert old in s, "stub anchor not found"
new = '''/* BC-250 experiment, see scripts/apply_gfx10_queue_reset.py */
static int bc250_queue_reset;
module_param(bc250_queue_reset, int, 0644);
MODULE_PARM_DESC(bc250_queue_reset, "BC-250: 1 = gfx9-style per-queue compute reset on gfx10");

uint64_t kgd_gfx_v10_hqd_get_pq_addr(struct amdgpu_device *adev,
				     uint32_t pipe_id, uint32_t queue_id,
				     uint32_t inst)
{
	uint32_t low, high;
	uint64_t queue_addr = 0;

	if (!bc250_queue_reset)
		return 0;

	acquire_queue(adev, pipe_id, queue_id);
	amdgpu_gfx_rlc_enter_safe_mode(adev, inst);

	if (!RREG32_SOC15(GC, 0, mmCP_HQD_ACTIVE))
		goto unlock_out;

	low = RREG32_SOC15(GC, 0, mmCP_HQD_PQ_BASE);
	high = RREG32_SOC15(GC, 0, mmCP_HQD_PQ_BASE_HI);

	/* only concerned with user queues. */
	if (!high)
		goto unlock_out;

	queue_addr = (((queue_addr | high) << 32) | low) << 8;

unlock_out:
	amdgpu_gfx_rlc_exit_safe_mode(adev, inst);
	release_queue(adev);

	return queue_addr;
}

/* assume queue acquired */
static int kgd_gfx_v10_hqd_dequeue_wait(struct amdgpu_device *adev,
					unsigned int utimeout)
{
	unsigned long end_jiffies = (utimeout * HZ / 1000) + jiffies;

	while (true) {
		uint32_t temp = RREG32_SOC15(GC, 0, mmCP_HQD_ACTIVE);

		if (!(temp & CP_HQD_ACTIVE__ACTIVE_MASK))
			return 0;

		if (time_after(jiffies, end_jiffies))
			return -ETIME;

		usleep_range(500, 1000);
	}
}

uint64_t kgd_gfx_v10_hqd_reset(struct amdgpu_device *adev,
			       uint32_t pipe_id, uint32_t queue_id,
			       uint32_t inst, unsigned int utimeout)
{
	uint32_t low, high, pipe_reset_data = 0;
	uint64_t queue_addr = 0;
	bool via_pipe = false;

	if (!bc250_queue_reset)
		return 0;

	acquire_queue(adev, pipe_id, queue_id);
	amdgpu_gfx_rlc_enter_safe_mode(adev, inst);

	if (!RREG32_SOC15(GC, 0, mmCP_HQD_ACTIVE))
		goto unlock_out;

	low = RREG32_SOC15(GC, 0, mmCP_HQD_PQ_BASE);
	high = RREG32_SOC15(GC, 0, mmCP_HQD_PQ_BASE_HI);

	/* only concerned with user queues. */
	if (!high)
		goto unlock_out;

	queue_addr = (((queue_addr | high) << 32) | low) << 8;

	dev_info(adev->dev, "BC250QRESET attempting queue reset pipe %u queue %u\\n",
		 pipe_id, queue_id);

	/* assume previous dequeue request issued will take affect after reset */
	WREG32_SOC15(GC, 0, mmSPI_COMPUTE_QUEUE_RESET, 0x1);

	if (!kgd_gfx_v10_hqd_dequeue_wait(adev, utimeout))
		goto unlock_out;

	via_pipe = true;
	dev_info(adev->dev, "BC250QRESET queue still active, attempting pipe reset pipe %u\\n",
		 pipe_id);

	pipe_reset_data = REG_SET_FIELD(pipe_reset_data, CP_MEC_CNTL, MEC_ME1_PIPE0_RESET, 1);
	pipe_reset_data = pipe_reset_data << pipe_id;

	WREG32_SOC15(GC, 0, mmCP_MEC_CNTL, pipe_reset_data);
	WREG32_SOC15(GC, 0, mmCP_MEC_CNTL, 0);

	if (kgd_gfx_v10_hqd_dequeue_wait(adev, utimeout))
		queue_addr = 0;

unlock_out:
	if (queue_addr || via_pipe)
		dev_info(adev->dev, "BC250QRESET pipe %u queue %u %s%s\\n", pipe_id, queue_id,
			 queue_addr ? "succeeded" : "failed", via_pipe ? " (pipe reset)" : "");
	amdgpu_gfx_rlc_exit_safe_mode(adev, inst);
	release_queue(adev);

	return queue_addr;
}
'''
s = s.replace(old, new, 1)
p.write_text(s)
print("patched", p)
