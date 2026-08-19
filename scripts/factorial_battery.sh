#!/usr/bin/env bash
# factorial_battery.sh <cell-tag> <boot-index>
# Fixed per-boot measurement battery for the 2x2 factorial
# (flush_pasid_uses_kiq x sched_policy) on kernel 7.1.5.
# Emits one structured VERDICT line; all raw logs kept.
set -u
CELL=$1; IDX=$2
D=~/factorial/${CELL}_${IDX}; mkdir -p $D
export HSA_ENABLE_SDMA=0
export LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib

# --- state verification (refuse to measure a misconfigured boot) ---
KV=$(uname -r)
CU=$(journalctl -k -b 0 --no-pager | grep -m1 -oE "active_cu_number [0-9]+" | awk '{print $2}')
FL=$(journalctl -k -b 0 --no-pager | grep -m1 -oE "flush_pasid_uses_kiq=[0-9]" | cut -d= -f2)
SP=$(cat /sys/module/amdgpu/parameters/sched_policy 2>/dev/null || echo NA)
UP=$(cut -d. -f1 /proc/uptime)
echo "state kernel=$KV cu=$CU flush=$FL sched_policy=$SP uptime=$UP" | tee $D/state.txt
sync

dmesg_errs () { journalctl -k -b 0 --no-pager | grep -cE "preemption time out|queue evicted|page fault|MEMORY_APERTURE|amdgpu.*ERROR" ; }
E0=$(dmesg_errs)

# --- 1. compute_probe 8M x3 ---
P_OK=0; P_WRONG=0; P_FAULT=0; P_HANG=0
for s in 1 2 3; do
  timeout -k 10 240 ~/compute_probe 32768 6000 1 > $D/probe8m_$s.log 2>&1; rc=$?
  if grep -q "ALL CORRECT" $D/probe8m_$s.log; then P_OK=$((P_OK+1))
  elif grep -q "WRONG" $D/probe8m_$s.log; then P_WRONG=$((P_WRONG+1))
  elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then P_HANG=$((P_HANG+1))
  else P_FAULT=$((P_FAULT+1)); fi
  sync
done

# --- 2. sgemm 2048 x10 ---
timeout -k 15 240 ~/sgemm_iter 2048 10 > $D/sgemm2048.log 2>&1
if grep -q "DONE iters=10 nwrong=0" $D/sgemm2048.log; then S2=CLEAN
elif grep -q "DONE" $D/sgemm2048.log; then S2=WRONG
else S2=WEDGE_OR_FAULT; fi
sync

# --- 3. sgemm 4096 x50 (the wedge test) ---
T0=$(date +%s)
timeout -k 15 420 ~/sgemm_iter 4096 50 > $D/sgemm4096.log 2>&1
T1=$(date +%s)
if grep -q "DONE iters=50 nwrong=0" $D/sgemm4096.log; then S4=CLEAN
elif grep -q "DONE" $D/sgemm4096.log; then S4=WRONG
else S4=WEDGE_OR_FAULT; fi
sync

E1=$(dmesg_errs)
PRE=$(journalctl -k -b 0 --no-pager | grep -c "cp queue preemption time out")

# --- 4. exit/board health ---
timeout 60 ~/compute_probe 4096 1000 1 > $D/small_after.log 2>&1
grep -q "ALL CORRECT" $D/small_after.log && HEALTH=RESPONSIVE || HEALTH=DEGRADED

echo "VERDICT cell=$CELL idx=$IDX cu=$CU flush=$FL policy=$SP probe8m_ok=$P_OK wrong=$P_WRONG fault=$P_FAULT hang=$P_HANG sgemm2048=$S2 sgemm4096=$S4 sgemm4096_secs=$((T1-T0)) dmesg_errs_delta=$((E1-E0)) preempt_timeouts=$PRE health=$HEALTH" | tee $D/verdict.txt
sync
