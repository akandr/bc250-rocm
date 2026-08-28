#!/usr/bin/env bash
# Does this board survive a GPU reset at all?
#
# Every fatal event on record ends with "GPU reset begin!" and no line saying it
# finished, followed by the board going away. That could mean the reset is fatal
# here, or it could mean the state the reset was called from was unrecoverable.
# Triggering one deliberately, from an idle GPU, separates those.
set -u
D=~/inv111
log () { echo "[$(date +%T)] $*" >> "$D/log"; sync; }
log "=== arm: $1, governor $(systemctl is-active oberon-governor), gpu idle"
log "  uptime before: $(cut -d. -f1 /proc/uptime)s"
# Reading this node is what triggers the reset; writing to it does nothing at all,
# which cost an hour once. The path is discovered rather than written out because
# debugfs exposes the device under both a minor number and its PCI address, and
# `find` reports only one of them, so hard-coding either is a guess. Failing loudly
# beats a silent no-op: the original of this script hard-coded a path and sent its
# error to a log nobody read.
N=$(sudo find /sys/kernel/debug/dri -maxdepth 2 -name amdgpu_gpu_recover | head -1)
log "  trigger node: ${N:-NOT FOUND}"
[ -n "$N" ] || { log "=== no trigger node, nothing was reset"; exit 1; }
sudo cat "$N" 2>>"$D/log"
log "  trigger returned $?"
sleep 20
log "  uptime after: $(cut -d. -f1 /proc/uptime)s"
log "  kernel: $(sudo dmesg | grep -icE "reset begin") reset-begin, $(sudo dmesg | grep -icE "reset.*succeed|recovered") succeeded"
sudo dmesg | grep -iE "reset|recover" | tail -6 >> "$D/log"
log "  governor now: $(systemctl is-active oberon-governor), restarts $(systemctl show oberon-governor -p NRestarts --value)"
log "=== done"
