#!/usr/bin/env bash
# override_trap_probe.sh - does HSA_OVERRIDE_GFX_VERSION silently zero a native kernel?
#
# INVESTIGATION.md carries a two-row table saying the bare compute probe, built
# for gfx1013, returns all 4,194,304 elements correct in about 2 seconds without
# the override and all of them zero in 0.1 ms under it. That table is a safety
# warning about a workaround people are still told to use, and until now it had
# no capture behind it: no probe log in this repository holds either timing.
#
# Both arms are the same binary on the same boot, run back to back, native
# gfx1013. The override arm is expected to fail; a run that succeeds under it is
# the interesting outcome and is what this is for.
set -u
D="${1:-$HOME/override-trap}"
mkdir -p "$D"
NB=16384        # 16384 blocks x 256 threads = 4194304 elements, the table's count
INNER=6000
{
  echo "kernel: $(uname -r)"
  echo "date: $(date -Is)"
  echo "cmdline: $(cat /proc/cmdline)"
  echo "probe: $(ls -l ~/compute_probe 2>/dev/null || echo MISSING)"
} > "$D/state.txt"

run() {  # name, then the environment assignment or empty
  local name="$1"; shift
  echo "=== $name ===" >> "$D/probe.out"
  ( set -x; env "$@" HSA_ENABLE_SDMA=0 timeout -k 10 300 ~/compute_probe "$NB" "$INNER" 1 ) \
    >> "$D/probe.out" 2>&1
  echo "rc=$?" >> "$D/probe.out"
}

: > "$D/probe.out"
run "no override" HSA_ENABLE_SDMA=0
run "HSA_OVERRIDE_GFX_VERSION=10.1.0" HSA_OVERRIDE_GFX_VERSION=10.1.0
# Count from the persistent journal, not dmesg. dmesg cannot see a fault from a
# boot that ended in a reset, since after a reboot it reports the boot that
# follows; the journal keeps both. Neither arm here is expected to reset, but a
# counter that is only correct when nothing goes wrong is the wrong counter.
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
echo "--- fault count, this boot ---" >> "$D/probe.out"
sudo journalctl -b 0 --no-pager 2>/dev/null | grep -cE "$PAT" >> "$D/probe.out" 2>&1
echo "--- fault count, previous boot ---" >> "$D/probe.out"
sudo journalctl -b -1 --no-pager 2>/dev/null | grep -cE "$PAT" >> "$D/probe.out" 2>&1
