#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Rebuild Fedora's rocm-runtime 7.1.1 (ROCr) with the queue-scratch scope-guard fix from rocm-systems
# PR #2850 and stage libhsa-runtime64 in /opt/bc250-rocm/lib64, ahead of the distribution copy, next to
# the native rocBLAS and the corrected comgr (scripts/install_rocm711_overrides.sh). Without it a context
# that crosses the KFD resident-memory limit dies with a segfault inside libhsa-runtime64; with it the
# runtime returns the error and llama.cpp prints "ROCm error: out of memory"
# (logs/rocr-queue-scratch-2026-09-18/). The system package is not touched; a dnf update of rocm-runtime
# does not replace the staged copy. Remove: sudo rm /opt/bc250-rocm/lib64/libhsa-runtime64.so.1*; sudo ldconfig
#
# Usage: build_rocr711_scratch_fix.sh [work dir]      (CPU-heavy, ten minutes on the board; GPU idle)
set -eu
W=${1:-$HOME/rocr-build}; HERE=$(cd "$(dirname "$0")/.." && pwd)
PATCH=$HERE/patches/rocr-guard-queue-scratch-release.patch
mkdir -p "$W" && cd "$W"
# dnf on this board must not touch mesa or the kernel (see the README's Fedora 44 notes): exclude them on every call
DNF="sudo dnf --exclude=mesa* --exclude=kernel* --setopt=install_weak_deps=False"
$DNF install -y rpm-build rpmdevtools dnf-plugins-core >/dev/null || true   # build tools only
[ -f rocm-runtime-7.1.1-*.src.rpm ] || dnf download --source rocm-runtime
srpm=$(ls rocm-runtime-7.1.1-*.src.rpm | head -1)
mkdir -p rpmbuild/{SOURCES,SPECS,BUILD,RPMS,SRPMS}
rpm2cpio "$srpm" | (cd rpmbuild/SOURCES && cpio -idm --quiet)
mv rpmbuild/SOURCES/rocm-runtime.spec rpmbuild/SPECS/ 2>/dev/null || true
cp "$PATCH" rpmbuild/SOURCES/0002-rocr-guard-queue-scratch-release.patch
python3 - rpmbuild/SPECS/rocm-runtime.spec <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
# add the patch after the last Patch: line; the spec's %autosetup -p3 applies it with the a/projects/... prefix
lines = s.split("\n"); idx = max(i for i, l in enumerate(lines) if l.startswith("Patch"))
lines.insert(idx + 1, "Patch:      0002-rocr-guard-queue-scratch-release.patch")
s = "\n".join(lines)
s = re.sub(r"^Release:\s*(\S+)", r"Release:    \1.bc250", s, count=1, flags=re.M)
# current rpm rejects this spec's out-of-order %changelog; the section is metadata only
if "%changelog" in s: s = s[:s.index("%changelog")]
open(p, "w").write(s); print("spec: patch added")
PY
$DNF builddep -y rpmbuild/SPECS/rocm-runtime.spec
rpmbuild -bb --define "_topdir $W/rpmbuild" rpmbuild/SPECS/rocm-runtime.spec
rpm=$(ls rpmbuild/RPMS/x86_64/rocm-runtime-7.1.1-*.x86_64.rpm | grep -v debug | head -1)
mkdir -p extract && (cd extract && rpm2cpio "../$rpm" | cpio -idm --quiet)
lib=$(find extract -name "libhsa-runtime64.so.1.*" -type f | head -1)
D=/opt/bc250-rocm/lib64
sudo mkdir -p $D && sudo install -m 0755 "$lib" $D/$(basename "$lib")
sudo ln -sfn $(basename "$lib") $D/libhsa-runtime64.so.1
grep -qx $D /etc/ld.so.conf.d/bc250-rocm.conf 2>/dev/null || echo $D | sudo tee /etc/ld.so.conf.d/bc250-rocm.conf >/dev/null
command -v restorecon >/dev/null && sudo restorecon -R $D
sudo ldconfig
ldconfig -p | grep "libhsa-runtime64.so.1 "     # the /opt/bc250-rocm entry must come first
