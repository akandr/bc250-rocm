#!/usr/bin/env bash
# Can a direct TLB invalidation replace the runlist rebuild, once it bypasses the KIQ?
#
# Runs on the workstation: cells can hang the board, so each cell is one ssh
# call and a dead board is power-cycled before the next. Netconsole runs for the
# whole sequence so a hang still leaves a trail.
#
# The discriminator is the one from logs/vmid-flush-2026-08-20: an 8B model load
# with mmap off, which aborts with no flush and completes with the runlist
# rebuild. Module from scripts/apply_tlb_reset_experiments.py.
#
# Usage: scripts/tlb_alt_ab.sh "<runlist>:<alt> ..."   e.g. "3:0 0:0 3:2 3:1 3:4 3:3"
#
# Fault counting here reads dmesg on a boot that is still up, and was left that way once the run
# was logged, since changing the counter would change what the log means. dmesg cannot see a
# fault from a run that ended by taking the board down; new work should use
# scripts/fault_count.sh, which reads the persistent journal.
set -u
CELLS=${1:-"3:0 0:0 3:2 3:1 3:4 3:3"}
BOARD=${BOARD:-bc250}
OUT=${OUT:-logs/tlb-alt-2026-09-15}
PORT=6969
LISTENER_IP=${LISTENER_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1)}
mkdir -p "$OUT"
log () { echo "[$(date +%T)] $*" | tee -a "$OUT/ab.log"; }
up () { ssh -o ConnectTimeout=10 -o BatchMode=yes "$BOARD" true 2>/dev/null; }

prep () {
  ssh -o ConnectTimeout=20 "$BOARD" "LISTENER_IP=$LISTENER_IP bash ~/netconsole_capture.sh >/dev/null 2>&1;
    for u in ollama.service signal-cli.service; do sudo systemctl stop \$u; done
    for t in claude-batch hw-watcher car-watcher claude-improve daily-digest plocate-updatedb dnf-makecache; do sudo systemctl stop \$t.timer 2>/dev/null; done" </dev/null >/dev/null 2>&1
}

recover () {
  log "  board down, power cycling"
  bash "$(dirname "$0")/bc250_power" off >/dev/null 2>&1; sleep 15
  bash "$(dirname "$0")/bc250_power" on >/dev/null 2>&1
  for i in $(seq 1 20); do up && break; sleep 15; done
  prep
}

pkill -f "nc -u -l $PORT" 2>/dev/null; sleep 1
(nc -u -l $PORT >> "$OUT/netconsole_ab.log" 2>/dev/null &)
up || recover
prep
n=0
for c in $CELLS; do
  n=$((n+1)); rl=${c%%:*}; alt=${c##*:}
  label="c${n}_rl${rl}_alt${alt}"
  ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 "$BOARD" "
    P=/sys/module/amdgpu/parameters
    echo MARK-$label | sudo tee /dev/kmsg >/dev/null
    echo $rl | sudo tee \$P/bc250_flush_by_runlist >/dev/null
    echo $alt | sudo tee \$P/bc250_tlb_alt >/dev/null
    sleep 1
    t0=\$(date +%s)
    env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib \
      timeout -k 20 300 ~/llama-master/build-hip/bin/llama-bench -m /opt/models/qwen3-8b-q8_0.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 1 > ~/s0915/$label.log 2>&1
    rc=\$?; t1=\$(date +%s)
    echo 3 | sudo tee \$P/bc250_flush_by_runlist >/dev/null; echo 0 | sudo tee \$P/bc250_tlb_alt >/dev/null
    tg=\$(grep -aoE 'tg64 \| +[0-9.]+' ~/s0915/$label.log | grep -oE '[0-9.]+\$')
    pp=\$(grep -aoE 'pp512 \| +[0-9.]+' ~/s0915/$label.log | grep -oE '[0-9.]+\$')
    err=\$(grep -aoE 'HSA_STATUS_ERROR[A-Z_]*|memory access fault|CUDA error' ~/s0915/$label.log | sort | uniq -c | tr '\n' ' ')
    alt_line=\$(sudo dmesg | grep BC250TLBALT | tail -1 | sed 's/.*BC250TLBALT//')
    faults=\$(sudo dmesg | grep -ciE 'page fault|GCVM_L2|preemption time|ring .* timeout')
    sync
    echo \"rc=\$rc pp512=\${pp:-NONE} tg64=\${tg:-NONE} wall=\$((t1-t0))s err=[\$err] faults_boot=\$faults alt:[\$alt_line]\"
  " </dev/null > "$OUT/$label.result" 2>&1
  if [ -s "$OUT/$label.result" ] && grep -q "^rc=" "$OUT/$label.result"; then
    log "$label: $(grep '^rc=' "$OUT/$label.result")"
    scp -q "$BOARD:s0915/$label.log" "$OUT/" 2>/dev/null
  else
    log "$label: NO RESULT (board lost during cell)"
    recover
  fi
done
pkill -f "nc -u -l $PORT" 2>/dev/null
log "done: $CELLS"
