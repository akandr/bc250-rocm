#!/usr/bin/env bash
# Install the amdgpu module carrying the KFD reset probe.
#
# The procedure is the one this repository documents and has been caught out by
# twice: strip the debug info, compress with --check=crc32 because the in-kernel
# decompressor rejects crc64, and run dracut, because the module that actually
# loads comes from the initramfs and not from /lib/modules. The previous module
# is kept so it can be put back.
set -eu
KREL=$(uname -r)
SRC=/home/akandr/k718/linux-7.1.8/drivers/gpu/drm/amd/amdgpu/amdgpu.ko
INST=/lib/modules/$KREL/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko.xz
D=~/inv119
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; }

test -f "$SRC" || { log "no built module at $SRC"; exit 1; }
sudo cp -n "$INST" "$D/amdgpu.ko.xz.backup" 2>/dev/null || true
log "=== backup of the installed module: $(ls -la $D/amdgpu.ko.xz.backup 2>/dev/null | awk '{print $5}') bytes"

cp "$SRC" /tmp/amdgpu.ko
strip --strip-debug /tmp/amdgpu.ko
log "  stripped to $(stat -c %s /tmp/amdgpu.ko) bytes"
rm -f /tmp/amdgpu.ko.xz
xz --check=crc32 /tmp/amdgpu.ko
log "  compressed to $(stat -c %s /tmp/amdgpu.ko.xz) bytes"

sudo cp /tmp/amdgpu.ko.xz "$INST"
sudo depmod -a "$KREL"
log "=== rebuilding the initramfs, without which the old module keeps loading"
sudo dracut -f --kver "$KREL" 2>&1 | tail -2 | tee -a "$D/log"
log "  initramfs: $(ls -la /boot/initramfs-$KREL.img | awk '{print $5}') bytes"
touch "$D/INSTALLED"; log "=== installed, reboot required"
