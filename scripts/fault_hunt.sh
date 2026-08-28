#!/usr/bin/env bash
# Hunt the fault that kills the board, and soak the current recipe while doing it.
#
# The first version of this only ran the fault-prone workload. But the recipe as
# it now stands, with the navi12 SDMA microcode and amdgpu.gpu_recovery=0, has
# never been soaked, and the previous soak predates both. The same hours can
# produce both results, so this adds a correctness gate every fifth round and
# records it, while keeping everything the hunt needs.
#
# Two things it must survive: the board rebooting under it, which systemd
# handles, and the evidence being invisible afterwards. Faults are read from
# both the current and the previous boot, because dmesg after a reset reports
# the boot that followed, and its ring buffer wraps on a long run besides.
#
# Stop it with: touch ~/faulthunt/STOP
set -u
D=~/faulthunt; mkdir -p "$D/hits"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
REF=8.9442
PAT="page fault \(src_id|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
# Faults raised on purpose by the probes must not be counted as natural events.
# They reproduce the signature exactly, so anything reading these archives later
# would have no way to tell them apart.
MINE="faultprobe|faultbusy|faultbig"
mine_only () {  # true when every faulting process named in $1 is one of ours
  local f=$1 names
  # amdgpu logs the faulting task as: Process <name> pid <n> thread <name> pid <n>
  # Matching "Process <word>" alone also catches "Process Core Dump Socket" and
  # PIDs from unrelated journal lines, which made this check always fail.
  names=$(grep -oE "Process [a-zA-Z0-9_-]+ pid " "$f" 2>/dev/null | awk '{print $2}' | sort -u)
  [ -z "$names" ] && return 1
  echo "$names" | grep -qvE "^($MINE)$" && return 1
  return 0
}
log () { echo "[$(date '+%F %T')] $*" >> "$D/log"; sync; }

boots=$(cat "$D/boots" 2>/dev/null || echo 0); echo $((boots+1)) > "$D/boots"
iters=$(cat "$D/iters" 2>/dev/null || echo 0)
log "=== resumed, boot $((boots+1)), $iters iterations so far, gpu_recovery=$(cat /sys/module/amdgpu/parameters/gpu_recovery), uptime $(cut -d. -f1 /proc/uptime)s"

prev=$(sudo journalctl -b -1 --no-pager 2>/dev/null | grep -cE "$PAT" || echo 0)
if [ "${prev:-0}" -gt 0 ] && [ ! -f "$D/hits/boot$boots.txt" ]; then
  log "  PREVIOUS BOOT FAULTED, archiving"
  sudo journalctl -b -1 --no-pager 2>/dev/null \
    | grep -E "$PAT|in page starting|PERMISSION_FAULTS|RW:|Faulty UTCL2|Process |GPU reset" | tail -50 > "$D/hits/boot$boots.txt"
  [ -f "$D/maps.live" ] && cp "$D/maps.live" "$D/hits/boot$boots.maps"
  # the mitigation question: did a reset actually happen, or only the fault?
  r=$(grep -c "GPU reset begin" "$D/hits/boot$boots.txt" || echo 0)
  if mine_only "$D/hits/boot$boots.txt"; then
    mv "$D/hits/boot$boots.txt" "$D/hits/boot$boots.deliberate.txt"
    log "  those faults were raised on purpose by the probes, not natural; archived as deliberate"
  else
    log "  fault lines archived; GPU reset begin lines in that boot: $r"
  fi
fi

while [ ! -f "$D/STOP" ]; do
  iters=$((iters+1)); echo $iters > "$D/iters"

  "$HIP/llama-bench" --version > /dev/null 2>&1
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 20 600 $HIP/llama-bench -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 \
    > "$D/run.log" 2>&1 &
  bp=$!
  for _ in 1 2 3 4 5 6 7 8; do
    sleep 20
    p=$(pgrep -n llama-bench 2>/dev/null) || continue
    [ -n "${p:-}" ] && sudo cat "/proc/$p/maps" > "$D/maps.tmp" 2>/dev/null && mv "$D/maps.tmp" "$D/maps.live"
    kill -0 $bp 2>/dev/null || break
  done
  wait $bp 2>/dev/null; rc=$?
  v=$(grep -aoE "tg8 @ d[0-9]+ \| +[0-9.]+" "$D/run.log" | grep -oE "[0-9.]+$")

  gate=""
  if [ $((iters % 5)) -eq 0 ]; then
    env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
      timeout -k 30 1800 $HIP/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa on \
      -c 4096 -f "$WIKI" --chunks 8 > "$D/gate.log" 2>&1
    gate=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/gate.log" | grep -oE "[0-9.]+$")
    [ "${gate:-x}" = "$REF" ] && gate="$gate ok" || gate="${gate:-FAIL} DEVIATES from $REF"
    echo "${gate}" >> "$D/gates"
  fi

  now=$(sudo journalctl -b 0 --no-pager 2>/dev/null | grep -cE "$PAT" || echo 0)
  if [ "${now:-0}" -gt "$(cat "$D/seen" 2>/dev/null || echo 0)" ]; then
    r=$(sudo journalctl -b 0 --no-pager 2>/dev/null | grep -c "GPU reset begin" || echo 0)
    log "  iteration $iters: FAULT IN THIS BOOT, board still up, GPU reset lines=$r (rc=$rc rate=${v:-none})"
    sudo journalctl -b 0 --no-pager 2>/dev/null \
      | grep -E "$PAT|in page starting|PERMISSION_FAULTS|RW:|Faulty UTCL2|Process " | tail -50 > "$D/hits/iter$iters.txt"
    cp "$D/maps.live" "$D/hits/iter$iters.maps" 2>/dev/null
    echo "$now" > "$D/seen"
  else
    [ -n "$gate" ] && log "  iteration $iters: clean (rate=${v:-none}) gate=$gate"
    [ -z "$gate" ] && [ $((iters % 25)) -eq 0 ] && log "  iteration $iters: clean (rate=${v:-none})"
  fi
done
log "=== stopped by request after $iters iterations"
