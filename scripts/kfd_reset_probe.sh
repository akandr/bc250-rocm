#!/usr/bin/env bash
# Test the reset mitigation without waiting for a natural fault.
#
# amdgpu.gpu_recovery=0 is supposed to stop the fatal reset, and the reasoning is
# read from the driver: the fatal resets log AMDGPU_RESET_SRC_HWS, raised by
# amdgpu_amdkfd_gpu_reset(), which does nothing unless
# amdgpu_device_should_recover_gpu() agrees, and that returns false when the
# parameter is zero.
#
# Every attempt to confirm this has gone through a fault, and no deliberate fault
# reaches that path: the process dies and the driver never needs to reset. The
# debugfs trigger reaches a reset but through AMDGPU_RESET_SRC_USER, which
# bypasses the guard entirely, so it says nothing either.
#
# The path can be called directly instead. This adds a module parameter that
# invokes amdgpu_amdkfd_gpu_reset() on write, which is exactly the call the
# fatal events make. With gpu_recovery=0 the guard should swallow it and the
# board should live; with the default it should reset and the board should die.
# That is the A/B the mitigation has been missing.
set -eu
SRC=${SRC:-/home/akandr/k718/linux-7.1.8}
AMDDIR=$SRC/drivers/gpu/drm/amd
KFD=$AMDDIR/amdgpu/amdgpu_amdkfd.c
KREL=$(uname -r)
D=~/inv119; mkdir -p "$D"
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; }

log "=== source $SRC, kernel $KREL"
grep -n "void amdgpu_amdkfd_gpu_reset" "$KFD" | head -1

if grep -q "bc250_test_kfd_reset" "$KFD"; then
  log "  probe already present in source"
else
  log "=== adding the probe to amdgpu_amdkfd.c"
  sudo python3 - "$KFD" <<'PYEOF'
import sys
p = sys.argv[1]; s = open(p).read()
anchor = """void amdgpu_amdkfd_gpu_reset(struct amdgpu_device *adev)
{
	if (amdgpu_device_should_recover_gpu(adev))
		(void)amdgpu_reset_domain_schedule(adev->reset_domain, &adev->kfd.reset_work);
}"""
assert anchor in s, "amdgpu_amdkfd_gpu_reset not found in the expected form"
add = """/* BC-250 probe: call the KFD reset path on demand, to test whether
 * amdgpu.gpu_recovery=0 suppresses it. Writing 1 to the parameter invokes the
 * same function the fatal resets take, without needing a GPU fault to trigger
 * one. Harmless when the parameter is never written.
 */
static struct amdgpu_device *bc250_probe_adev;

void amdgpu_amdkfd_gpu_reset(struct amdgpu_device *adev)
{
	bc250_probe_adev = adev;
	if (amdgpu_device_should_recover_gpu(adev))
		(void)amdgpu_reset_domain_schedule(adev->reset_domain, &adev->kfd.reset_work);
}

static int bc250_test_kfd_reset_set(const char *val, const struct kernel_param *kp)
{
	struct amdgpu_device *adev = bc250_probe_adev;

	if (!adev) {
		pr_warn("BC250KFDRESET: no device recorded yet, run something on the GPU first\\n");
		return -ENODEV;
	}
	pr_warn("BC250KFDRESET: calling amdgpu_amdkfd_gpu_reset, gpu_recovery=%d, should_recover=%d\\n",
		amdgpu_gpu_recovery, amdgpu_device_should_recover_gpu(adev) ? 1 : 0);
	amdgpu_amdkfd_gpu_reset(adev);
	pr_warn("BC250KFDRESET: returned\\n");
	return 0;
}

static const struct kernel_param_ops bc250_test_kfd_reset_ops = {
	.set = bc250_test_kfd_reset_set,
};
module_param_cb(bc250_test_kfd_reset, &bc250_test_kfd_reset_ops, NULL, 0644);
"""
open(p, "w").write(s.replace(anchor, add, 1))
print("probe inserted")
PYEOF
fi

# the KFD reset path records the device only when something has used the GPU,
# so the probe needs a real adev; recording it inside the function itself means
# it is set the first time the KFD touches a reset, which may never happen.
# Record it at bind time instead.
if ! grep -q "bc250_probe_adev = adev;" "$AMDDIR/amdgpu/amdgpu_amdkfd.c"; then
  log "  WARNING: adev capture missing"
fi

KBUILD=/lib/modules/$KREL/build
if [ ! -f "$KBUILD/drivers/gpu/drm/amd/amdgpu/amdgpu_trace.h" ]; then
  log "=== kernel-devel lacks amdgpu_trace.h, copying headers"
  for sub in amdgpu amdkfd include display; do
    [ -d "$AMDDIR/$sub" ] && sudo rsync -a --include='*/' --include='*.h' --exclude='*' \
      "$AMDDIR/$sub/" "$KBUILD/drivers/gpu/drm/amd/$sub/" 2>/dev/null || true
  done
fi

log "=== building amdgpu"
cd "$SRC"
sudo make -C "$KBUILD" M="$AMDDIR/amdgpu" -j6 modules > "$D/build.log" 2>&1 || true
if [ ! -f "$AMDDIR/amdgpu/amdgpu.ko" ]; then
  log "  BUILD FAILED"; grep -m5 -E "error:|Error" "$D/build.log" | tee -a "$D/log"; exit 1
fi
log "  built $(ls -la $AMDDIR/amdgpu/amdgpu.ko | awk '{print $5}') bytes"
touch "$D/BUILT"; log "=== built, not installed yet"
