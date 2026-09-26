#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# CLR rebuild, third pass: both HIP patches (Thread::Release guard, capture-stream null check).
set -u
O=~/clr-src
log() { echo "[$(date +%T)] $*" | tee -a $O/log3; }
: > $O/log3
cd $O || exit 1
cd x/clr && patch -p1 --dry-run < ~/0002-hip-graph-capture-null-stream.patch > /dev/null && log "patch 2 applies to the 7.1.1 tree" || { log "patch 2 does NOT apply"; exit 1; }
cd $O
cp ~/0002-hip-graph-capture-null-stream.patch ~/rpmbuild-clr/SOURCES/
python3 - <<'PY'
s = open("/home/akandr/rpmbuild-clr/SPECS/rocclr.spec").read()
if "Patch100:" not in s:
    s = s.replace("Patch99:    0001-rocclr-hostqueue-thread-release-null-vdev.patch\n",
                  "Patch99:    0001-rocclr-hostqueue-thread-release-null-vdev.patch\nPatch100:   0002-hip-graph-capture-null-stream.patch\n")
    s = s.replace("Release:    3%{?dist}.bc250", "Release:    3%{?dist}.bc250b") if "Release:    3%{?dist}.bc250" in s else s
    open("/home/akandr/rpmbuild-clr/SPECS/rocclr.spec", "w").write(s); print("spec: patch 2 added")
else: print("spec: already has patch 2")
PY
grep -n "^Patch\|^Release" ~/rpmbuild-clr/SPECS/rocclr.spec | tee -a $O/log3
log "rpmbuild start"
nice -n 10 rpmbuild -bb --nodeps --define "_topdir $HOME/rpmbuild-clr" --define "_smp_mflags -j6" ~/rpmbuild-clr/SPECS/rocclr.spec > $O/rpmbuild3.log 2>&1 && log "rpmbuild ok" || { log "rpmbuild FAILED"; grep -m8 -iE "error|błąd|patch" $O/rpmbuild3.log | tee -a $O/log3; }
rpm=$(ls -t ~/rpmbuild-clr/RPMS/x86_64/rocm-hip-7.1.1-*.x86_64.rpm 2>/dev/null | grep -v debug | head -1)
log "rpm: $rpm"
rm -rf $O/clr-extract3 && mkdir -p $O/clr-extract3 $O/hiplib3 && (cd $O/clr-extract3 && rpm2cpio "$rpm" | cpio -idm --quiet 2>/dev/null)
cp -a $(find $O/clr-extract3 -name "libamdhip64.so*") $O/hiplib3/ && ls -la $O/hiplib3 | tee -a $O/log3
md5sum $O/hiplib3/libamdhip64.so.7.1.52802 $O/hiplib/libamdhip64.so.7.1.52802 | cut -c1-12 | tr '\n' ' ' | tee -a $O/log3; echo | tee -a $O/log3
echo "CLR3 DONE" >> $O/log3
