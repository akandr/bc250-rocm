#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# CLR rebuild, second attempt: the first stopped at three missing build tools (fdupes, perl,
# perl-generators) that only serve packaging macros, and dnf on this board is blocked by an unrelated
# broken kernel-devel entry. So: the %fdupes line dropped from the spec, rpmbuild --nodeps, everything
# else as before. Waits for the fa-decode chain (CPU-heavy, no GPU measurement may overlap).
set -u
O=~/clr-src
log() { echo "[$(date +%T)] $*" | tee -a $O/log2; }
until grep -qE "CHAIN DONE|CHAIN FAILED" ~/fa-decode/log 2>/dev/null; do sleep 60; done
: > $O/log2
cd $O || exit 1
which perl >/dev/null && log "perl: $(perl -e 'print $^V')" || log "perl missing"
mkdir -p ~/rpmbuild-clr/{SOURCES,SPECS,BUILD,RPMS,SRPMS}
cp clr-7.1.1.tar.gz hip-7.1.1.tar.gz *.patch ~/rpmbuild-clr/SOURCES/ 2>/dev/null
cp ~/0001-rocclr-hostqueue-thread-release-null-vdev.patch ~/rpmbuild-clr/SOURCES/
python3 - <<'PY'
import re
s = open("/home/akandr/clr-src/rocclr.spec").read()
lines = s.split("\n"); idx = max(i for i, l in enumerate(lines) if re.match(r"Patch\d*:", l))
lines.insert(idx + 1, "Patch99:    0001-rocclr-hostqueue-thread-release-null-vdev.patch")
lines = [l for l in lines if not l.strip().startswith("%fdupes")]
s = "\n".join(lines)
s = re.sub(r"^Release:\s*(\S+)", r"Release:    \1.bc250", s, count=1, flags=re.M)
if "%changelog" in s: s = s[:s.index("%changelog")] + "%changelog\n* Fri Sep 19 2026 BC-250 repository - 7.1.1-3.bc250\n- HostQueue::Thread::Release: tolerate a failed Init (null virtual device)\n"
open("/home/akandr/rpmbuild-clr/SPECS/rocclr.spec", "w").write(s)
print("spec: patch added, %fdupes dropped, changelog entry")
PY
log "rpmbuild start (--nodeps)"
nice -n 10 rpmbuild -bb --nodeps --define "_topdir $HOME/rpmbuild-clr" --define "_smp_mflags -j6" ~/rpmbuild-clr/SPECS/rocclr.spec > $O/rpmbuild2.log 2>&1 && log "rpmbuild ok" || { log "rpmbuild FAILED"; grep -m10 -iE "error|błąd|nie ma|No such|patch|FAILED" $O/rpmbuild2.log | tee -a $O/log2; }
ls -la ~/rpmbuild-clr/RPMS/x86_64/ 2>/dev/null | tee -a $O/log2
rpm=$(ls ~/rpmbuild-clr/RPMS/x86_64/rocclr-7.1.1-*.x86_64.rpm 2>/dev/null | grep -v debug | head -1)
if [ -n "$rpm" ]; then
  rm -rf $O/clr-extract && mkdir -p $O/clr-extract && (cd $O/clr-extract && rpm2cpio "$rpm" | cpio -idm --quiet)
  mkdir -p $O/hiplib && cp -a $(find $O/clr-extract -name "libamdhip64.so*") $O/hiplib/ && ls -la $O/hiplib | tee -a $O/log2
  log "staged in $O/hiplib (not installed)"
fi
echo "CLR2 DONE" >> $O/log2
