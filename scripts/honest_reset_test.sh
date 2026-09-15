#!/usr/bin/env bash
# Board test for the two patches in patches/amdgpu/.
#
# Runs on the workstation. Needs the module from scripts/apply_tlb_reset_experiments.py.
#
#  A. Recovery default. Boots once without amdgpu.gpu_recovery=0 on the command line,
#     then asks the probe what amdgpu_device_should_recover_gpu() decides. Stock
#     upstream returns true for Cyan Skillfish at the default; with the stale RAS
#     early return out of the way it should log "GPU recovery disabled." and 0.
#
#  B. Honest reset. With amdgpu.bc250_honest_reset=1 a reset that has no
#     implementation returns -EOPNOTSUPP. Each trial reboots first, arms netconsole,
#     triggers a reset through debugfs (which bypasses the recovery guard), then
#     records whether the host is still reachable and whether a GPU job still runs.
#     Arms: MODE1 (the default method here) and MODE2.
#
# The command line is restored at the end, and checked.
#
# Fault counting here reads dmesg on a boot that is still up, and was left that way once the run
# was logged, since changing the counter would change what the log means. dmesg cannot see a
# fault from a run that ended by taking the board down; new work should use
# scripts/fault_count.sh, which reads the persistent journal.
# Its kernel counts are of "Queue preemption failed" lines and recovery refusals, not of page
# faults, and are read from dmesg on a boot still up after the trigger. A page fault that did not
# wedge a queue would not be counted. New work should use scripts/fault_count.sh.
set -u
BOARD=${BOARD:-bc250}
OUT=${OUT:-logs/reset-honest-2026-09-15}
PORT=6969
LISTENER_IP=${LISTENER_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1)}
K=/boot/vmlinuz-7.1.8-100.fc43.x86_64
mkdir -p "$OUT"
log () { echo "[$(date +%T)] $*" | tee -a "$OUT/log"; }
up () { ssh -o ConnectTimeout=8 -o BatchMode=yes "$BOARD" true 2>/dev/null; }
waitup () { for i in $(seq 1 40); do up && return 0; sleep 10; done; return 1; }
powercycle () { bash "$(dirname "$0")/bc250_power" off >/dev/null 2>&1; sleep 15; bash "$(dirname "$0")/bc250_power" on >/dev/null 2>&1; waitup; }
bootid () { ssh -o ConnectTimeout=8 "$BOARD" 'cat /proc/sys/kernel/random/boot_id' </dev/null 2>/dev/null; }
# A reboot on a wedged GPU can stall in shutdown with ssh still answering, and a trial would then
# run on the old boot without anyone noticing, so the boot id has to change or the power is cut.
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
smoke () { # a small GPU job; prints OK or FAIL
  ssh -o ConnectTimeout=15 "$BOARD" 'LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib timeout 120 ~/llama-master/build-hip/bin/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 64 -n 16 -r 1 2>&1 | grep -q "tg16" && echo OK || echo FAIL' </dev/null 2>/dev/null || echo UNREACHABLE
}

up || powercycle
ssh "$BOARD" "sudo grubby --info=$K | grep ^args" </dev/null > "$OUT/cmdline_before.txt"
log "=== A. recovery default: removing amdgpu.gpu_recovery=0 for this test"
ssh "$BOARD" "sudo grubby --update-kernel=$K --remove-args=amdgpu.gpu_recovery=0" </dev/null
reboot_board; prep
log "  gpu_recovery=$(ssh "$BOARD" 'cat /sys/module/amdgpu/parameters/gpu_recovery' </dev/null)"
log "  smoke job (records the device for the probe): $(smoke)"
ssh "$BOARD" 'echo 1 | sudo tee /sys/module/amdgpu/parameters/bc250_should_recover_probe >/dev/null; sudo dmesg | grep -E "BC250RECOVER|GPU recovery disabled" | tail -3' </dev/null | tee "$OUT/A_probe.txt" | while read -r l; do log "  $l"; done

log "=== B. honest reset"
t=0
for arm in "mode1:bc250_honest_reset=1" "mode1:bc250_honest_reset=1" "mode2:reset_method=3 bc250_honest_reset=1" "mode2:reset_method=3 bc250_honest_reset=1"; do
  t=$((t+1)); name=${arm%%:*}; params=${arm#*:}; label="B${t}_${name}"
  reboot_board; prep
  pkill -f "nc -u -l $PORT" 2>/dev/null; sleep 1
  (nc -u -l $PORT > "$OUT/$label.netconsole.log" 2>/dev/null &); sleep 2
  ssh "$BOARD" "echo MARK-$label | sudo tee /dev/kmsg >/dev/null" </dev/null
  for kv in $params; do
    ssh "$BOARD" "echo ${kv#*=} | sudo tee /sys/module/amdgpu/parameters/${kv%%=*} >/dev/null" </dev/null
  done
  log "$label: params [$params], pre-reset smoke $(smoke)"
  ssh -o ConnectTimeout=15 "$BOARD" 'N=$(sudo find /sys/kernel/debug/dri -maxdepth 2 -name amdgpu_gpu_recover | head -1); sudo timeout 60 cat "$N"' </dev/null > "$OUT/$label.trigger.txt" 2>&1 &
  trig=$!
  sleep 70; kill $trig 2>/dev/null
  if up; then
    ssh "$BOARD" 'sudo dmesg | grep -E "GPU reset|reset|BC250|EOPNOTSUPP|not implemented|failed|resume" | tail -25' </dev/null > "$OUT/$label.dmesg.txt" 2>&1
    log "$label: host ALIVE after reset; post-reset smoke $(smoke); trigger returned: $(tr '\n' ' ' < "$OUT/$label.trigger.txt")"
  else
    log "$label: host UNREACHABLE after reset"
  fi
  pkill -f "nc -u -l $PORT" 2>/dev/null
  log "$label: netconsole tail: $(grep -v RDSEED "$OUT/$label.netconsole.log" | tail -4 | cut -c1-120 | tr '\n' '|')"
done

log "=== C. per-queue compute reset on gfx10 (scripts/apply_gfx10_queue_reset.py)"
# Trigger: the PASID-filtered rebuild (bc250_tlb_alt=5) under a --no-mmap perplexity load puts
# a queue into the preemption-failed state within seconds (logs/tlb-alt-2026-09-15/ round 4).
# A merely long dispatch does not: the scheduler preempts it cleanly (hang_probe, same day).
# reset_queues_on_hws_hang() only runs when gpu_recovery is non-zero, which it is here.
t=0
for arm in 0 1 0 1; do
  t=$((t+1)); label="C${t}_queue_reset${arm}"
  reboot_board; prep
  pkill -f "nc -u -l $PORT" 2>/dev/null; sleep 1
  (nc -u -l $PORT > "$OUT/$label.netconsole.log" 2>/dev/null &); sleep 2
  ssh "$BOARD" "echo MARK-$label | sudo tee /dev/kmsg >/dev/null; echo $arm | sudo tee /sys/module/amdgpu/parameters/bc250_queue_reset >/dev/null" </dev/null
  log "$label: pre smoke $(smoke)"
  ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=12 "$BOARD" '
    P=/sys/module/amdgpu/parameters
    echo 5 | sudo tee $P/bc250_tlb_alt >/dev/null
    env GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib \
      timeout -k 10 120 ~/llama-master/build-hip/bin/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 > ~/s0915/trigger.log 2>&1
    echo "trigger rc=$?"
    echo 0 | sudo tee $P/bc250_tlb_alt >/dev/null
    sleep 10' </dev/null > "$OUT/$label.trigger.txt" 2>&1
  if up; then
    s1=$(smoke); sleep 20; s2=$(smoke)
    ssh "$BOARD" 'sudo dmesg | grep -E "MARK|BC250QRESET|preemption failed|GPU recovery disabled|GPU reset|hung|runlist rebuild flush failed|ENOTRECOVERABLE|reset" | grep -v "reset reason\|factory-reset"' </dev/null > "$OUT/$label.dmesg.txt" 2>&1
    log "$label: host ALIVE; $(tr '\n' ' ' < "$OUT/$label.trigger.txt"); preempt_fail=$(grep -c "preemption failed" "$OUT/$label.dmesg.txt") qreset_lines=$(grep -c BC250QRESET "$OUT/$label.dmesg.txt") recovery_disabled=$(grep -c "GPU recovery disabled" "$OUT/$label.dmesg.txt"); smoke after trigger: $s1, 20 s later: $s2"
  else
    log "$label: host UNREACHABLE after trigger"
  fi
  pkill -f "nc -u -l $PORT" 2>/dev/null
done

log "=== restoring the command line"
up || powercycle
ssh "$BOARD" "sudo grubby --update-kernel=$K --args=amdgpu.gpu_recovery=0; sudo grubby --info=$K | grep ^args" </dev/null | tee "$OUT/cmdline_after.txt"
reboot_board
log "  after restore: gpu_recovery=$(ssh "$BOARD" 'cat /sys/module/amdgpu/parameters/gpu_recovery' </dev/null)"
log "done"
