#!/bin/bash
# Build amdgpu.ko with flush_pasid_uses_kiq=false for gfx1013 (ROCm#6313 patch)
# via the Duggan module-only pipeline. Runs ON THE BOARD.
set -euo pipefail
SRC=/usr/src/linux-6.18.9
AMDDIR=$SRC/drivers/gpu/drm/amd
GMC=$AMDDIR/amdgpu/gmc_v10_0.c
KREL=$(uname -r)
INST=/lib/modules/$KREL/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko.xz

echo "=== pre-checks"
test -d "$SRC" || { echo "no kernel source at $SRC"; exit 1; }
grep -n "flush_pasid_uses_kiq" "$GMC"
# confirm 40cu patch still present in source (sanity per memory)
grep -qn "bc250_cc_write_mode" $AMDDIR/amdgpu/gfx_v10_0.c && echo "40cu patch present in source"

echo "=== patching gmc_v10_0.c"
if grep -q "flush_pasid_uses_kiq = false" "$GMC"; then
  echo "already patched"
else
  sudo sed -i 's/adev->gmc.flush_pasid_uses_kiq = !amdgpu_emu_mode;/adev->gmc.flush_pasid_uses_kiq = false; \/* BC-250 ROCm#6313 *\//' "$GMC"
  grep -n "flush_pasid_uses_kiq" "$GMC"
fi

echo "=== building amdgpu module only"
cd $SRC
sudo make -C /lib/modules/$KREL/build M=$SRC/drivers/gpu/drm/amd/amdgpu -j4 modules 2>&1 | tail -5

MOD=$SRC/drivers/gpu/drm/amd/amdgpu/amdgpu.ko
test -f $MOD || { echo "BUILD FAILED"; exit 1; }
sudo strip --strip-debug $MOD

echo "=== installing (backup first)"
if [ ! -f $INST.prepasidfix-backup ]; then
  sudo cp -v $INST $INST.prepasidfix-backup
fi
# IMPORTANT: kernel module xz MUST use --check=crc32. The xz default (crc64) loads via
# userspace modprobe but fails the IN-KERNEL decompressor (finit_module from initramfs)
# with "decompression failed with status 6" -> GPU never comes up after a dracut -f.
sudo sh -c "xz -c -f --check=crc32 --lzma2=preset=6,dict=1MiB $MOD > $INST"
sudo depmod $KREL
xz -t $INST && echo "module xz integrity OK (crc32)" || { echo "XZ INTEGRITY FAIL"; exit 1; }
echo "=== done; reboot to activate. Restore: cp $INST.prepasidfix-backup $INST && depmod && reboot"
