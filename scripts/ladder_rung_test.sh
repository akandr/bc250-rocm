#!/usr/bin/env bash
# ladder_rung_test.sh - per-rung battery. Run AFTER booting a rung kernel.
# Answers per rung: (i) 40-CU unlock fired? (ii) HWS sustained GEMM survives
# (the 6.18 wedge)? (iii) compute correct (ppl gate)? (iv) churn-fault behavior.
# Ends by arming a one-shot boot back to 7.1.5 (does not reboot itself).
set -u
KREL=$(uname -r)
D=~/ladder-results/$KREL; mkdir -p $D
H=~/llama-master/build-hip/bin
export HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$HOME/rocBLAS/build/release/rocblas-install/lib
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a $D/rung.log; sync; }

log "=== RUNG $KREL"
log "cmdline: $(cat /proc/cmdline | grep -oE "amdgpu[^ ]*" | tr "\n" " ")"
cu=$(rocminfo 2>/dev/null | grep -m1 -iA1 "Compute Unit" | tail -1 | grep -oE "[0-9]+")
unlock=$(sudo dmesg | grep -c "bc250-40cu-enable" || true)
flushline=$(sudo dmesg | grep -m1 -iE "flush_pasid|bc250.*flush" || true)
log "(i) CU=$cu unlock_lines=$unlock flush='$flushline'"

log "(ii) sustained GEMM x20 N=4096 (wedge test)"
timeout -k 15 900 ~/sgemm_iter 4096 20 > $D/sgemm.log 2>&1
rc=$?
wedge=$(sudo dmesg | grep -c "cp queue preemption time out" || true)
log "(ii) rc=$rc preemption_timeouts=$wedge last='$(tail -1 $D/sgemm.log | cut -c1-60)'"

log "(iii) ppl gate chunks2 ctx1024 (expect ~9.83)"
timeout -k 15 600 $H/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 1024 -f ~/wiki.test.raw --chunks 2 > $D/ppl.log 2>&1
log "(iii) rc=$? $(grep -oE "Final estimate.*" $D/ppl.log || grep -m1 -oE "ROCm error.*" $D/ppl.log)"

log "(iv) TBO churn x1 (fault behavior, param=1 runlist v1)"
timeout -k 10 900 $H/test-backend-ops perf -o MUL_MAT -b ROCm0 > $D/tbo.log 2>&1
log "(iv) rc=$? faults=$(grep -ac "Memory access fault" $D/tbo.log)"

sudo dmesg > $D/dmesg-full.txt
log "RUNG $KREL DONE; arming boot back to 7.1.5"
sudo grub2-reboot 0
log "run: sudo reboot"
echo done > $D/DONE
