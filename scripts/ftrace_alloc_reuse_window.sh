#!/usr/bin/env bash
# ftrace_alloc_reuse_window.sh - capture the unmap-versus-dispatch window around an
# alloc-reuse fault. TBO churn faults within seconds (3/3); run it under
# ftrace and freeze the buffer the moment the process exits, then classify
# by the process log and dmesg (pattern: "gfxhub.*page fault").
# INDUCES FAULTS - run when nothing else needs the GPU.
set -u
D=~/inv28; mkdir -p $D
T=/sys/kernel/tracing
B=~/llama-master/build-hip/bin/test-backend-ops
export HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$HOME/rocBLAS/build/release/rocblas-install/lib
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a $D/inv28.log; sync; }

EVENTS="amdgpu_vm_bo_unmap amdgpu_vm_bo_map amdgpu_vm_update_ptes amdgpu_vm_set_ptes amdgpu_vm_copy_ptes amdgpu_vm_flush amdgpu_pasid_freed amdgpu_iv"

sudo bash -c "
  echo 0 > $T/tracing_on
  echo > $T/trace
  echo 131072 > $T/buffer_size_kb
  echo mono > $T/trace_clock
  for e in $EVENTS; do echo 1 > $T/events/amdgpu/\$e/enable 2>/dev/null; done
"
log "ftrace armed (v2): $EVENTS"

for run in 1 2 3; do
  dm_start=$(sudo dmesg | wc -l)
  sudo bash -c "echo > $T/trace; echo 1 > $T/tracing_on; echo '=== INV28 RUN $run START ===' > $T/trace_marker"
  log "run $run: TBO churn under trace"
  timeout -k 10 900 $B perf -o MUL_MAT -b ROCm0 > $D/tbo_r$run.log 2>&1
  rc=$?
  sudo bash -c "echo 0 > $T/tracing_on"
  ufault=$(grep -ac "Memory access fault" $D/tbo_r$run.log)
  sudo dmesg | tail -n +$((dm_start+1)) > $D/dmesg_r$run.txt
  kfault=$(grep -c "gfxhub.*page fault" $D/dmesg_r$run.txt)
  if [ "$ufault" -gt 0 ] || [ "$kfault" -gt 0 ]; then
    sudo cat $T/trace > $D/trace_r$run.txt
    sudo chown akandr $D/trace_r$run.txt 2>/dev/null
    log "run $run: FAULT rc=$rc ufault=$ufault kfault_lines=$kfault trace=$(wc -l < $D/trace_r$run.txt) lines"
  else
    log "run $run: NO FAULT rc=$rc"
  fi
  sleep 5
done

sudo bash -c "
  echo 0 > $T/tracing_on
  for e in $EVENTS; do echo 0 > $T/events/amdgpu/\$e/enable 2>/dev/null; done
"
log "INV28 v2 capture done"
echo done > $D/DONE2
