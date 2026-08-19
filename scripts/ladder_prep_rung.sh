#!/usr/bin/env bash
# ladder_prep_rung.sh <version> <release> - prepare one kernel rung of the ladder.
# Installs the koji kernel, fetches+patches its amdgpu source, builds the
# patched module, installs with crc32 xz, dracut for THAT kernel, fixes boot
# args. Does NOT reboot. Patch set = prod-equivalent: 40cu unlock +
# flush_pasid_kiq param + runlist flush v1 (unmap ioctl). Anchor drift on
# older trees -> report and stop; port by hand.
set -euo pipefail
V=$1; R=$2
KREL="$V-$R.x86_64"
LD=~/koji-ladder/$V-$R
SRCD=~/ladder-src
log () { echo "[$(date +%H:%M:%S)] [$V] $*"; }

log "install rung kernel RPMs"
sudo dnf install -y $LD/kernel-*.rpm --setopt=installonly_limit=0 2>&1 | tail -2 || {
  rpm -q kernel-core-$V-$R.x86_64 > /dev/null || { log "INSTALL FAILED"; exit 1; }
  log "already installed"
}

log "fetch + extract source"
mkdir -p $SRCD && cd $SRCD
SRPM=kernel-$V-$R.src.rpm
[ -s $SRPM ] || curl -sfO https://kojipkgs.fedoraproject.org/packages/kernel/$V/$R/src/$SRPM
if [ ! -d linux-$V ]; then
  rpm2cpio $SRPM | cpio -id --quiet "linux-$V.tar.xz" "patch-*-redhat.patch"
  tar xf linux-$V.tar.xz
  cd linux-$V
  patch -p1 --quiet < ../patch-*-redhat.patch 2>/dev/null || patch -p1 --quiet < $(ls ../patch-*-redhat.patch | head -1)
  cd ..
  rm -f linux-$V.tar.xz patch-*-redhat.patch
fi
S=$SRCD/linux-$V

log "apply 40cu unlock patch"
cd $S
if grep -q bc250_cc_write_mode drivers/gpu/drm/amd/amdgpu/gfx_v10_0.c; then
  log "  already applied"
else
  patch -p1 --forward < ~/bc250-40cu-unlock/patch/bc250-40cu-amdgpu.patch || { log "40CU PATCH DRIFT - port by hand"; exit 2; }
fi

log "apply flush_pasid_kiq param"
python3 - "$S" <<'PYEOF'
import sys, re, os
S = sys.argv[1]
p = os.path.join(S, "drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c")
s = open(p).read()
if "bc250_flush_pasid_kiq" in s:
    print("  already applied"); sys.exit(0)
# param declaration after the includes-adjacent MODULE area: reuse 715 approach:
# force flush_pasid_uses_kiq from a module param, default 1 (stock)
anchor = "adev->gmc.flush_pasid_uses_kiq = !amdgpu_emu_mode;"
if anchor not in s:
    print("  FLUSHPARAM ANCHOR DRIFT"); sys.exit(2)
decl = ("static int bc250_flush_pasid_kiq = 1;\n"
        "module_param(bc250_flush_pasid_kiq, int, 0444);\n"
        "MODULE_PARM_DESC(bc250_flush_pasid_kiq, \"BC-250: 0=flush_pasid_uses_kiq=false (ROCm#6313)\");\n\n")
first_func = s.index("static ")
s = s[:first_func] + decl + s[first_func:]
s = s.replace(anchor, "adev->gmc.flush_pasid_uses_kiq = bc250_flush_pasid_kiq ? !amdgpu_emu_mode : false; /* BC-250 */", 1)
open(p, "w").write(s)
print("  applied")
PYEOF
rc=$?; [ $rc -eq 2 ] && exit 2

log "apply runlist flush"
# Earlier revisions of this script rewrote the apply script with sed to replace a
# hardcoded tree path, and patched its anchors for kernels where kfd_flush_tlb
# takes two arguments. Both of those defects were fixed in the apply script
# itself, which now takes the amdkfd directory as an argument, validates it, and
# matches either signature. So this is a direct call.
python3 "$(dirname "$0")/apply_runlist_flush.py" "$S/drivers/gpu/drm/amd/amdkfd" || exit 2
python3 "$(dirname "$0")/apply_svmflush_generic.py" "$S/drivers/gpu/drm/amd/amdkfd" || exit 2
rc=$?; [ $rc -ne 0 ] && { log "RUNLIST PATCH FAILED ($rc) - port by hand"; exit 2; }

log "build module against $KREL devel"
test -d /usr/src/kernels/$KREL || { log "no devel tree for $KREL"; exit 1; }
cd $S
make -C /usr/src/kernels/$KREL M=$S/drivers/gpu/drm/amd/amdgpu -j"$(nproc)" modules 2>&1 | tail -2
MOD=$S/drivers/gpu/drm/amd/amdgpu/amdgpu.ko; test -f $MOD
strip --strip-debug $MOD
INST=/lib/modules/$KREL/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko.xz
sudo cp -n $INST $INST.stock-backup 2>/dev/null || true
sudo sh -c "xz -c -f --check=crc32 --lzma2=preset=6,dict=1MiB $MOD > $INST"
sudo depmod $KREL
xz -t $INST

log "dracut for $KREL + boot args"
sudo dracut -f --kver $KREL
sudo grubby --update-kernel /boot/vmlinuz-$KREL \
  --args "amdgpu.bc250_cc_write_mode=3 amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=1" \
  --remove-args "amdgpu.sched_policy" 2>/dev/null || true
sudo grubby --info /boot/vmlinuz-$KREL | grep -E "^(index|args)"
log "RUNG $V-$R PREPARED (boot with: sudo grub2-reboot <index> && reboot)"
