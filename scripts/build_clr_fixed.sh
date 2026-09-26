#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Rebuild Fedora's rocclr 7.1.1 (libamdhip64) with the HostQueue::Thread::Release null guard and extract
# the library without installing it. CPU-heavy (the whole HIP runtime); waits for gdn-chain2 so it does not
# heat the package during a GPU measurement. Then the D=256 flash-attention tile row sweep (also CPU-only).
set -u
O=~/clr-src; L=/opt/bc250-rocm/lib64
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
until grep -qE "CHAIN DONE|CHAIN FAILED" ~/gdn-chain2/log 2>/dev/null; do sleep 60; done
: > $O/log
cd $O || exit 1
mkdir -p ~/rpmbuild-clr/{SOURCES,SPECS,BUILD,RPMS,SRPMS}
cp clr-7.1.1.tar.gz hip-7.1.1.tar.gz *.patch ~/rpmbuild-clr/SOURCES/ 2>/dev/null
cp ~/0001-rocclr-hostqueue-thread-release-null-vdev.patch ~/rpmbuild-clr/SOURCES/
python3 - <<'PY'
import re
s = open("/home/akandr/clr-src/rocclr.spec").read()
lines = s.split("\n"); idx = max(i for i, l in enumerate(lines) if re.match(r"Patch\d*:", l))
lines.insert(idx + 1, "Patch99:    0001-rocclr-hostqueue-thread-release-null-vdev.patch")
s = "\n".join(lines)
s = re.sub(r"^Release:\s*(\S+)", r"Release:    \1.bc250", s, count=1, flags=re.M)
if "%changelog" in s: s = s[:s.index("%changelog")]
open("/home/akandr/rpmbuild-clr/SPECS/rocclr.spec", "w").write(s)
print("spec: patch added")
PY
grep -n "^Patch\|^Release\|%autosetup\|%setup\|%prep" -A2 ~/rpmbuild-clr/SPECS/rocclr.spec | head -20 | tee -a $O/log
log "rpmbuild start"
nice -n 10 rpmbuild -bb --define "_topdir $HOME/rpmbuild-clr" --define "_smp_mflags -j6" ~/rpmbuild-clr/SPECS/rocclr.spec > $O/rpmbuild.log 2>&1 && log "rpmbuild ok" || { log "rpmbuild FAILED"; grep -m8 -iE "error|missing|needed by|patch" $O/rpmbuild.log | tee -a $O/log; }
ls -la ~/rpmbuild-clr/RPMS/x86_64/ 2>/dev/null | tee -a $O/log
rpm=$(ls ~/rpmbuild-clr/RPMS/x86_64/rocclr-7.1.1-*.x86_64.rpm 2>/dev/null | grep -v debug | head -1)
if [ -n "$rpm" ]; then
  rm -rf $O/clr-extract && mkdir -p $O/clr-extract && (cd $O/clr-extract && rpm2cpio "$rpm" | cpio -idm --quiet)
  find $O/clr-extract -name "libamdhip64.so*" | tee -a $O/log
  mkdir -p $O/hiplib && cp -a $(find $O/clr-extract -name "libamdhip64.so*") $O/hiplib/
  log "staged in $O/hiplib (not installed)"
fi
echo "CLR BUILD DONE" >> $O/log
# D=256 tile rows: candidates that might reach zero spill
cd ~/llama-master
~/fa_row_sweep.sh 256 "32:512:2:32:64" "32:512:2:32:32" "32:1024:2:32:64" "32:1024:1:32:128" "32:256:2:32:64" "16:512:2:32:64" "16:512:2:32:32" "16:1024:2:32:64" "16:256:3:32:64" "16:256:2:32:32" > ~/fa-remainder/sweep256.log 2>&1
echo "SWEEP DONE" >> ~/fa-remainder/sweep256.log
echo "ALL DONE" >> $O/log
