#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Rebuild Fedora's rocm-runtime 7.1.1 with the queue-scratch scope-guard fix (rocm-systems PR #2850) and
# stage the resulting libhsa-runtime64 in /opt/bc250-rocm/lib64, where the native rocBLAS and the corrected
# comgr already live and ld.so prefers. Does not touch the system package. CPU-heavy: run with the GPU idle.
set -u
cd ~/rocr-src || exit 1
O=~/rocr-src/log; : > $O
log() { echo "[$(date +%T)] $*" | tee -a $O; }
mkdir -p ~/rpmbuild/{SOURCES,SPECS,BUILD,RPMS,SRPMS}
cp rocr-runtime-7.1.1.tar.gz 0001-hsakmt-bump-vgpr-count-for-gfx1151-1807-1986.patch 0002-rocr-guard-queue-scratch-release.patch ~/rpmbuild/SOURCES/
python3 - <<'PY'
s = open("rocm-runtime.spec").read()
old = "Patch:      0001-hsakmt-bump-vgpr-count-for-gfx1151-1807-1986.patch\n"
new = old + "Patch:      0002-rocr-guard-queue-scratch-release.patch\n"
assert s.count(old) == 1
s = s.replace(old, new, 1)
s = s.replace("Release:", "Release:    6.bc250%{?dist}\n#Release:", 1)
# rpm rejects this spec's out-of-order %changelog as an error; the section is metadata only
s = s[:s.index("%changelog")] if "%changelog" in s else s
open("rocm-runtime-bc250.spec", "w").write(s)
print("spec: patch 2 added")
PY
grep -n "^Patch\|^Release" rocm-runtime-bc250.spec | tee -a $O
cp rocm-runtime-bc250.spec ~/rpmbuild/SPECS/
log "rpmbuild start"
rpmbuild -bb --define "_topdir $HOME/rpmbuild" ~/rpmbuild/SPECS/rocm-runtime-bc250.spec > ~/rocr-src/rpmbuild.log 2>&1 && log "rpmbuild ok" || { log "rpmbuild FAILED (see rpmbuild.log)"; tail -20 ~/rocr-src/rpmbuild.log | tee -a $O; exit 1; }
ls -la ~/rpmbuild/RPMS/x86_64/ | tee -a $O
rpm=$(ls ~/rpmbuild/RPMS/x86_64/rocm-runtime-7.1.1-*.x86_64.rpm | grep -v debug | head -1)
mkdir -p ~/rocr-src/extract && cd ~/rocr-src/extract && rpm2cpio "$rpm" | cpio -idm --quiet
find . -name "libhsa-runtime64.so*" | tee -a $O
log "built; not installed. To stage: sudo cp extract/usr/lib64/libhsa-runtime64.so.1.18.0 /opt/bc250-rocm/lib64/ && sudo ln -sf libhsa-runtime64.so.1.18.0 /opt/bc250-rocm/lib64/libhsa-runtime64.so.1 && sudo ldconfig"
