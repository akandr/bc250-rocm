# Testing a newer kernel (7.1.5) with the 40-CU unlock

Goal: check whether the correctness defect and the compute-queue wedge are a kernel
regression by booting a much newer kernel than the 6.18.9 baseline. Fedora 43 shipped
kernel 7.1.5-100.fc43 in `updates`, which is newer than the ROCm 7.1 / kernel 7.0 stack
community members report on in ROCm/ROCm#6313.

BC-250 display support and `bc250_cc_write_mode` are upstream from ~6.18, so a newer
kernel needs only the duggasco 40-CU unlock patched in (not the display backport that
blocked the older-kernel test). Flush is left stock (`flush_pasid_uses_kiq = true`), so
this replicates the 40-CU / flush-true baseline config on the new kernel.

Steps (run on the board):

```bash
# 1. install the newer kernel + its devel package (keeps 6.18.9 as fallback)
sudo dnf install -y kernel-7.1.5-100.fc43 kernel-devel-7.1.5-100.fc43

# 2. get the full kernel source (kernel-devel ships no driver .c for 7.1.5)
sudo dnf download --source kernel-7.1.5-100.fc43 --destdir ~/k715
cd ~/k715 && rpm2cpio kernel-7.1.5-100.fc43.src.rpm | cpio -idm && tar xf linux-7.1.5.tar.xz

# 3. apply only the 40-CU unlock (leave flush stock/true)
cd linux-7.1.5
patch -p1 < ~/bc250-40cu-unlock/patch/bc250-40cu-amdgpu.patch   # applies with fuzz

# 4. the out-of-tree module build needs the amdgpu source in the kernel-devel tree
#    (7.1.5 kernel-devel ships only Kconfig/Makefile), so mirror it there
KD=/usr/src/kernels/7.1.5-100.fc43.x86_64
sudo cp -r drivers/gpu/drm/amd/* $KD/drivers/gpu/drm/amd/
sudo make -C $KD M=drivers/gpu/drm/amd/amdgpu -j6 modules

# 5. install with crc32 xz (the in-kernel initramfs decompressor rejects the xz default crc64)
KO=$KD/drivers/gpu/drm/amd/amdgpu/amdgpu.ko
INST=/lib/modules/7.1.5-100.fc43.x86_64/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko.xz
sudo strip --strip-debug $KO
sudo cp $INST $INST.orig-stock
sudo sh -c "xz -c -f --check=crc32 --lzma2=preset=6,dict=1MiB $KO > $INST"
sudo xz -t $INST && sudo depmod 7.1.5-100.fc43.x86_64

# 6. rebuild the 7.1.5 initramfs (it is the authoritative copy of amdgpu at boot)
sudo dracut -f --kver 7.1.5-100.fc43.x86_64

# 7. boot 7.1.5 (the kernel cmdline already carries amdgpu.bc250_cc_write_mode=3), then verify
#    active_cu_number 40 before trusting the setup.
```

Result: boots at 40 CU, display and Vulkan work, and both the correctness defect and the
wedge reproduce. See `compute_probe_fresh_samples.log`, `compute_probe_first_sweep.log`,
and `dmesg_faults.log` in this directory. Sampler: `scripts/probe_kernel_sweep.sh`.
