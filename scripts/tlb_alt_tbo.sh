#!/usr/bin/env bash
# Stale-TLB alternatives against the churn discriminator, one fresh boot per boot-group.
#
# Written after scripts/tlb_alt_ab.sh ran its cells back to back and the first
# MMIO arm broke queue creation (-62) for the rest of that boot, leaving the later
# cells unreadable. Here every group of cells starts from a reboot.
#
# Workload: test-backend-ops perf -o MUL_MAT, the allocation-churn workload from
# logs/svm-flush-2026-08/, which faulted within about 14 s with the map-side flush
# off and ran ten minutes clean with it on. A cell is capped at CAP seconds; a run
# still going at the cap counts as survived-to-cap.
#
# Usage: scripts/tlb_alt_tbo.sh "<group> <group> ..." where a group is cells joined
# by '+', each cell <runlist>:<alt>. Example: "3:0+1:0 3:0+3:4 3:0+3:3"
#
# Fault counting here reads dmesg on a boot that is still up, and was left that way once the run
# was logged, since changing the counter would change what the log means. dmesg cannot see a
# fault from a run that ended by taking the board down; new work should use
# scripts/fault_count.sh, which reads the persistent journal.
set -u
GROUPS_=${1:?groups}
CAP=${CAP:-150}
BOARD=${BOARD:-bc250}
OUT=${OUT:-logs/tlb-alt-2026-09-15/tbo}
PORT=6969
LISTENER_IP=${LISTENER_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1)}
mkdir -p "$OUT"
log () { echo "[$(date +%T)] $*" | tee -a "$OUT/tbo.log"; }
up () { ssh -o ConnectTimeout=10 -o BatchMode=yes "$BOARD" true 2>/dev/null; }
waitup () { for i in $(seq 1 40); do up && return 0; sleep 10; done; return 1; }
prep () {
  ssh -o ConnectTimeout=20 "$BOARD" "LISTENER_IP=$LISTENER_IP bash ~/netconsole_capture.sh >/dev/null 2>&1;
    for u in ollama.service signal-cli.service; do sudo systemctl stop \$u; done
    for t in claude-batch hw-watcher car-watcher claude-improve daily-digest plocate-updatedb dnf-makecache; do sudo systemctl stop \$t.timer 2>/dev/null; done; mkdir -p ~/s0915/tbo" </dev/null >/dev/null 2>&1
}
fresh () {
  if up; then
    ssh -o ConnectTimeout=10 "$BOARD" 'sudo systemctl reboot' </dev/null >/dev/null 2>&1
    sleep 40
  fi
  if ! waitup; then
    log "  not back after reboot, power cycling"
    bash "$(dirname "$0")/bc250_power" off >/dev/null 2>&1; sleep 15
    bash "$(dirname "$0")/bc250_power" on >/dev/null 2>&1; waitup
  fi
  prep
  log "  fresh boot: $(ssh "$BOARD" 'uptime -p; cat /sys/module/amdgpu/parameters/bc250_tlb_alt' </dev/null | tr '\n' ' ')"
}

pkill -f "nc -u -l $PORT" 2>/dev/null; sleep 1
(nc -u -l $PORT >> "$OUT/netconsole_tbo.log" 2>/dev/null &)
g=0
for grp in $GROUPS_; do
  g=$((g+1)); fresh
  for c in ${grp//+/ }; do
    rl=${c%%:*}; alt=${c##*:}; label="g${g}_rl${rl}_alt${alt}"
    ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 "$BOARD" "
      P=/sys/module/amdgpu/parameters
      echo MARK-$label | sudo tee /dev/kmsg >/dev/null
      echo $rl | sudo tee \$P/bc250_flush_by_runlist >/dev/null; echo $alt | sudo tee \$P/bc250_tlb_alt >/dev/null
      n0=\$(sudo dmesg | grep -c BC250TLBALT)
      t0=\$(date +%s)
      env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib \
        timeout -k 10 $CAP ~/llama-master/build-hip/bin/test-backend-ops perf -o MUL_MAT -b ROCm0 > ~/s0915/tbo/$label.log 2>&1
      rc=\$?; t1=\$(date +%s)
      echo 3 | sudo tee \$P/bc250_flush_by_runlist >/dev/null; echo 0 | sudo tee \$P/bc250_tlb_alt >/dev/null
      f=\$(grep -ac 'Memory access fault' ~/s0915/tbo/$label.log)
      alt_line=\$(sudo dmesg | grep BC250TLBALT | tail -n +\$((n0+1)) | tail -1 | sed 's/.*BC250TLBALT//')
      k=\$(sudo dmesg | grep -ciE 'page fault|preemption|create queue .* failed|ring .* timeout|Fence fallback timer expired on ring sdma')
      sync
      echo \"rc=\$rc fault=\$f wall=\$((t1-t0))s kernel_events_boot=\$k alt:[\$alt_line]\"
    " </dev/null > "$OUT/$label.result" 2>&1
    if grep -q "^rc=" "$OUT/$label.result"; then
      log "$label: $(grep '^rc=' "$OUT/$label.result")"
    else
      log "$label: NO RESULT (board lost)"; break
    fi
  done
done
pkill -f "nc -u -l $PORT" 2>/dev/null
ssh -o ConnectTimeout=10 "$BOARD" 'echo 3 | sudo tee /sys/module/amdgpu/parameters/bc250_flush_by_runlist; echo 0 | sudo tee /sys/module/amdgpu/parameters/bc250_tlb_alt' </dev/null >/dev/null 2>&1
log "done: $GROUPS_"
