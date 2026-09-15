#!/usr/bin/env bash
# Can a wedged GPU be brought back by unbinding and rebinding amdgpu, instead of a reboot?
#
# After a fault under gpu_recovery=0 the GPU stays unusable until reboot
# (logs/fault-usability-2026-08-24/), and neither a full reset (none exists for this chip) nor a
# gfx9-style queue reset (logs/reset-honest-2026-09-15/) recovers it. Reloading the driver was
# listed as the obvious untested candidate. This tests it.
#
# Runs on the workstation. Needs the experiment module (bc250_tlb_alt=5 is the wedge trigger:
# a --no-mmap perplexity load under the PASID-filtered rebuild puts a queue into the
# preemption-failed state within seconds, logs/tlb-alt-2026-09-15/ round 4).
#
#   R0  healthy GPU: unbind, bind, smoke. Whether a rebind works at all on this APU.
#   R1, R2  wedge, confirm the smoke job fails, unbind, bind, smoke.
#
# Each trial starts from its own boot (boot id checked). Only oberon-governor holds the DRM
# render node on this board, so it is stopped before the unbind; there is no framebuffer console.
#
# State checks read dmesg on a boot that is still up, and what they count is "preemption failed"
# lines, as evidence that the wedge took hold, not page faults; a page fault that did not wedge a
# queue would not be counted. See scripts/fault_count.sh for faults that end in a lost board.
set -u
BOARD=${BOARD:-bc250}
OUT=${OUT:-logs/rebind-recovery-2026-09-15}
PORT=6969
LISTENER_IP=${LISTENER_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1)}
DEV=0000:01:00.0
mkdir -p "$OUT"
log () { echo "[$(date +%T)] $*" | tee -a "$OUT/log"; }
up () { ssh -o ConnectTimeout=8 -o BatchMode=yes "$BOARD" true 2>/dev/null; }
waitup () { for i in $(seq 1 40); do up && return 0; sleep 10; done; return 1; }
powercycle () { bash "$(dirname "$0")/bc250_power" off >/dev/null 2>&1; sleep 15; bash "$(dirname "$0")/bc250_power" on >/dev/null 2>&1; waitup; }
bootid () { ssh -o ConnectTimeout=8 "$BOARD" 'cat /proc/sys/kernel/random/boot_id' </dev/null 2>/dev/null; }
reboot_board () {
  local old; old=$(bootid)
  if [ -n "$old" ]; then ssh "$BOARD" 'sudo systemctl reboot' </dev/null >/dev/null 2>&1; sleep 45; fi
  for i in $(seq 1 12); do b=$(bootid); [ -n "$b" ] && [ "$b" != "$old" ] && return 0; sleep 10; done
  log "  reboot did not produce a new boot, power cycling"; powercycle
}
prep () {
  ssh "$BOARD" "LISTENER_IP=$LISTENER_IP bash ~/netconsole_capture.sh >/dev/null 2>&1;
    for u in ollama.service signal-cli.service; do sudo systemctl stop \$u; done
    for t in claude-batch hw-watcher car-watcher claude-improve daily-digest plocate-updatedb dnf-makecache; do sudo systemctl stop \$t.timer 2>/dev/null; done" </dev/null >/dev/null 2>&1
}
smoke () {
  ssh -o ConnectTimeout=15 "$BOARD" 'LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib timeout 120 ~/llama-master/build-hip/bin/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 64 -n 16 -r 1 2>&1 | grep -aoE "tg16 \| +[0-9.]+" | grep -oE "[0-9.]+$" || echo FAIL' </dev/null 2>/dev/null || echo UNREACHABLE
}

for trial in R0 R1 R2; do
  reboot_board; prep
  pkill -f "nc -u -l $PORT" 2>/dev/null; sleep 1
  (nc -u -l $PORT > "$OUT/$trial.netconsole.log" 2>/dev/null &); sleep 2
  ssh "$BOARD" "echo MARK-$trial | sudo tee /dev/kmsg >/dev/null" </dev/null
  log "$trial: boot smoke tg16=$(smoke)"

  if [ "$trial" != R0 ]; then
    ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=12 "$BOARD" '
      P=/sys/module/amdgpu/parameters
      echo 5 | sudo tee $P/bc250_tlb_alt >/dev/null
      env GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib \
        timeout -k 10 120 ~/llama-master/build-hip/bin/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 > ~/s0915/rebind_trigger.log 2>&1
      echo 0 | sudo tee $P/bc250_tlb_alt >/dev/null' </dev/null
    log "$trial: wedged? preemption failures=$(ssh "$BOARD" 'sudo dmesg | grep -c "preemption failed"' </dev/null), smoke tg16=$(smoke)"
  fi

  ssh "$BOARD" "echo MARK-$trial-unbind | sudo tee /dev/kmsg >/dev/null; sudo systemctl stop oberon-governor; sudo pkill -f 'llama-|test-backend' ; sleep 2" </dev/null
  ub=$(ssh -o ServerAliveInterval=10 -o ServerAliveCountMax=12 "$BOARD" "t0=\$(date +%s); sudo timeout 90 sh -c 'echo $DEV > /sys/bus/pci/drivers/amdgpu/unbind'; echo \"rc=\$? \$((\$(date +%s)-t0))s\"" </dev/null 2>&1 | tail -1)
  log "$trial: unbind $ub; host $(up && echo up || echo DOWN)"
  if up; then
    bd=$(ssh -o ServerAliveInterval=10 -o ServerAliveCountMax=18 "$BOARD" "echo MARK-$trial-bind | sudo tee /dev/kmsg >/dev/null; t0=\$(date +%s); sudo timeout 150 sh -c 'echo $DEV > /sys/bus/pci/drivers/amdgpu/bind'; echo \"rc=\$? \$((\$(date +%s)-t0))s\"" </dev/null 2>&1 | tail -1)
    log "$trial: bind $bd; host $(up && echo up || echo DOWN)"
  fi
  if up; then
    ssh "$BOARD" "sudo dmesg | sed -n '/MARK-$trial-unbind/,\$p'" </dev/null > "$OUT/$trial.dmesg_after_unbind.txt" 2>&1
    log "$trial: after rebind: dri=[$(ssh "$BOARD" 'ls /dev/dri | tr "\n" " "' </dev/null)] cu_lines=$(grep -c bc250-40cu-enable "$OUT/$trial.dmesg_after_unbind.txt") errors=$(grep -ciE 'error|failed|timeout' "$OUT/$trial.dmesg_after_unbind.txt") smoke tg16=$(smoke)"
  else
    log "$trial: host lost; netconsole tail: $(grep -v RDSEED "$OUT/$trial.netconsole.log" | tail -3 | cut -c1-120 | tr '\n' '|')"
  fi
  pkill -f "nc -u -l $PORT" 2>/dev/null
done
reboot_board
log "done"
