#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Install the two ROCm 7.1.1 fixes for gfx1013 system-wide on Fedora 44, ahead of the distribution
# libraries, so no LD_LIBRARY_PATH is needed:
#   - native gfx1013 rocBLAS 7.1.1 (scripts/apply_gfx1013_rocblas711.py + build_rocblas711_gfx1013.sh)
#   - libamd_comgr.so.3 with the gfx10 VGPR count corrected (scripts/fix_comgr_gfx10_vgprs.py)
# Libraries go to /opt/bc250-rocm/lib64, listed in /etc/ld.so.conf.d, which the dynamic loader
# searches before /usr/lib64. rocBLAS finds its Tensile library relative to itself.
#
# Usage: install_rocm711_overrides.sh <rocblas install prefix> <fixed libamd_comgr.so.3>
# Remove: sudo rm -r /opt/bc250-rocm /etc/ld.so.conf.d/bc250-rocm.conf && sudo ldconfig
# A dnf update of rocblas or rocm-comgr does not touch these copies; rebuild or re-patch after one.
set -eu
RB=$1; CG=$2
D=/opt/bc250-rocm/lib64
sudo mkdir -p $D
sudo cp -a "$RB"/lib64/librocblas.so* $D/
sudo rm -rf $D/rocblas && sudo cp -a "$RB"/lib64/rocblas $D/rocblas
sudo install -m 0755 "$CG" $D/libamd_comgr.so.3
echo $D | sudo tee /etc/ld.so.conf.d/bc250-rocm.conf >/dev/null
command -v restorecon >/dev/null && sudo restorecon -R /opt/bc250-rocm /etc/ld.so.conf.d/bc250-rocm.conf
sudo ldconfig
ldconfig -p | grep -E "librocblas.so.5 |libamd_comgr.so.3 "
