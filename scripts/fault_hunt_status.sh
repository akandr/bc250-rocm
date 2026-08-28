#!/usr/bin/env bash
# One-screen status of the fault-usability hunt, for the loop to read each cycle.
#
# Reports what has been covered and, more importantly, whether anything has been
# caught. A hunt that is merely running is not a result, and the distinction that
# matters is between "no fault yet" and "a fault arrived and the battery ran".
set -u
BOARD=${BOARD:-bc250}
if ! ssh -o ConnectTimeout=15 "$BOARD" 'echo up' </dev/null 2>/dev/null | grep -q up; then
	echo "board not answering ssh; it may be mid-reboot, wedged, or powered off"
	exit 2
fi
ssh -o ConnectTimeout=25 "$BOARD" '
D=~/faulthunt2
echo "service:    $(systemctl is-active faulthunt2)"
echo "uptime:     $(cut -d. -f1 /proc/uptime)s, service start $(cat $D/starts 2>/dev/null || cat $D/boots 2>/dev/null || echo 0)"
echo "iterations: $(cat $D/iters 2>/dev/null || echo 0)"
echo "recovery:   gpu_recovery=$(cat /sys/module/amdgpu/parameters/gpu_recovery)"
h=$(ls $D/hits/*.usability 2>/dev/null | grep -v healthy-control | wc -l)
f=$(ls $D/hits/iter*.txt 2>/dev/null | wc -l)
echo "faults:     $f caught, $h battery run(s) beyond the control"
echo "--- last log lines"
tail -6 $D/log 2>/dev/null
if [ "$f" -gt 0 ]; then
  echo "--- FAULT CAUGHT, batteries:"
  for b in $D/hits/*.usability; do
    case $b in *healthy-control*) continue;; esac
    echo "=== $b"; cat "$b"
  done
fi' </dev/null 2>&1
