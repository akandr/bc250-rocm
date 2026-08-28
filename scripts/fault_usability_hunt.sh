#!/usr/bin/env bash
# Catch a real GPU fault under amdgpu.gpu_recovery=0 and find out what the
# machine is like afterwards.
#
# The parameter is confirmed to stop the reset: called directly, the guard
# returns and the board stays up. That is not the same as the machine being
# usable. Under the parameter a fault leaves a dead process and, in principle, a
# wedged GPU on a live host, and nobody has looked at the "usable" half. It is
# the largest gap in the advice this repository gives other owners.
#
# The previous hunt could not answer it. On a fault it logged "board still up"
# and carried on, which conflates the loop continuing with the machine being
# healthy: the next iteration launches a fresh process, and if that hangs the
# hunt just looks slow. This one stops on the fault and interrogates the state
# while it is still there.
#
# Everything in the battery has a timeout, because a hang is an answer here and
# must not be allowed to look like a long wait. Each step records what it found
# and whether it returned at all.
#
# Faults are rare. The last hunt saw none in 255 iterations across 16 boots, so
# this rotates workloads rather than repeating one, on the reasoning that the
# fault that started this arrived during an eight-hour soak rotating large models
# and not during the one shape the previous hunt repeated.
#
# Stop it with: touch ~/faulthunt2/STOP
set -u
D=~/faulthunt2; mkdir -p "$D/hits"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
# The gate the battery runs is four chunks, not the eight the older hunts used,
# so it needs its own reference. Measured on this configuration with the board
# healthy, immediately before the hunt started, because a post-fault value with
# nothing to compare it against says nothing.
REF4=8.4240
PAT="page fault \(src_id|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
log () { echo "[$(date '+%F %T')] $*" >> "$D/log"; sync; }

# Faults are counted from the persistent journal, never from dmesg: after a reset
# dmesg reports the boot that followed, and its ring buffer wraps on a long run.
faults_now () { sudo journalctl -b 0 --no-pager 2>/dev/null | grep -cE "$PAT" || echo 0; }

# The battery. Every question the phrase "is the machine usable" actually means,
# asked in increasing order of ambition so the first thing that hangs says how
# far usability survived.
usability () {
  # Two statements, not one: bash expands every word of a `local` before it
  # assigns any of them, so `local tag=$1 out=...$tag...` reads tag while it is
  # still unset and dies under set -u. That would have fired on the first real
  # fault, after however many days of waiting for one.
  local tag=$1
  local out="$D/hits/$tag.usability"
  {
    echo "=== battery $tag, $(date '+%F %T'), uptime $(cut -d. -f1 /proc/uptime)s"
    echo "--- did a reset happen at all (the parameter says it should not)"
    sudo journalctl -b 0 --no-pager 2>/dev/null | grep -cE "GPU reset begin" \
      | sed 's/^/GPU reset begin lines this boot: /'
    echo "--- does the kernel still see the device"
    timeout -k 5 30 lspci -s 01:00.0 -vv 2>&1 | head -5 || echo "TIMED OUT or failed"
    # rocminfo is not installed on this board, so enumeration is asked of the
    # runtime that actually matters here. A healthy board names the device and
    # reports free memory; silence is the failure, which is why the control run
    # below is kept alongside.
    echo "--- does the runtime still enumerate the agent and report free memory"
    timeout -k 5 90 env LD_LIBRARY_PATH=$L $HIP/llama-bench --list-devices 2>&1 \
      | grep -E "found [0-9]+ ROCm|ROCm0:" || echo "TIMED OUT or enumerated nothing"
    # 65536 blocks, not 1048576: the larger size takes about seventeen minutes
    # and a shorter timeout truncated it to its banner, which reads exactly like
    # a hang. This size returns a verdict line in about a minute.
    echo "--- can a fresh process allocate and run a kernel, and is the answer right"
    timeout -k 10 300 env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L ~/compute_probe 65536 2>&1 \
      | grep -E "^RESULT:|^iter 8:" || echo "TIMED OUT or produced no verdict"
    echo "--- does a real workload still run and still produce the right answer"
    timeout -k 20 900 env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
      $HIP/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa on -c 4096 -f "$WIKI" --chunks 4 2>&1 \
      | grep -aoE "Final estimate: PPL = [0-9.]+" || echo "TIMED OUT or produced no estimate"
    echo "--- healthy reference for that gate is PPL = $REF4"
    echo "=== battery $tag ends, uptime $(cut -d. -f1 /proc/uptime)s"
  } > "$out" 2>&1
  log "  battery written to hits/$tag.usability"
}

# Run the battery once and exit. The control taken with the board healthy has to
# be the same code as the one taken after a fault, or the comparison is between
# two instruments rather than two states.
if [ "${1:-}" = "--battery" ]; then
	usability "${2:?usage: --battery <tag>}"
	cat "$D/hits/${2}.usability"
	exit 0
fi

# The expectation under this parameter is that the host survives a fault. If it
# does not, journald loses the tail, exactly as it did for every reset before
# netconsole was used. Arming it at every boot costs nothing and is the only
# instrument that has ever worked on a machine that stops writing to disk. It is
# best effort: if the listener is not reachable the hunt still runs.
if [ -x "$HOME/netconsole_capture.sh" ] || [ -f "$HOME/netconsole_capture.sh" ]; then
	bash "$HOME/netconsole_capture.sh" >> "$D/netconsole_arm.log" 2>&1 \
		&& log "  netconsole armed" || log "  netconsole not armed, continuing without it"
fi

# This counts service starts, not boots. systemd restarts the unit whenever the
# script exits or is replaced, so calling it "boot" overstated what it measured:
# the log read "boot 4" on a board that had not rebooted since "boot 3". The
# actual boot is reported beside it from uptime.
starts=$(cat "$D/starts" 2>/dev/null || cat "$D/boots" 2>/dev/null || echo 0)
starts=$((starts+1)); echo $starts > "$D/starts"
iters=$(cat "$D/iters" 2>/dev/null || echo 0)
log "=== resumed, service start $starts, $iters iterations so far, gpu_recovery=$(cat /sys/module/amdgpu/parameters/gpu_recovery), uptime $(cut -d. -f1 /proc/uptime)s"

# A fault in the previous boot means the host did not survive it, which is itself
# the answer to the question this script exists to ask.
prev=$(sudo journalctl -b -1 --no-pager 2>/dev/null | grep -cE "$PAT" || echo 0)
if [ "${prev:-0}" -gt 0 ] && [ ! -f "$D/hits/start$starts.txt" ]; then
  log "  PREVIOUS BOOT FAULTED AND THE HOST DID NOT SURVIVE IT, archiving"
  sudo journalctl -b -1 --no-pager 2>/dev/null \
    | grep -E "$PAT|in page starting|PERMISSION_FAULTS|RW:|Faulty UTCL2|Process |GPU reset" \
    | tail -60 > "$D/hits/start$starts.txt"
fi

# Baseline the journal count once per boot, not once per service start. Setting
# it on every start means a restart during a degraded boot re-baselines to the
# already-elevated count and the fault becomes undetectable, which is exactly
# what happened on 24 August: a restart to fix a logging bug erased the delta
# that the detection depended on.
bootid=$(cat /proc/sys/kernel/random/boot_id)
if [ "$(cat "$D/seen_bootid" 2>/dev/null)" != "$bootid" ]; then
	echo "$bootid" > "$D/seen_bootid"
	echo "$(faults_now)" > "$D/seen"
	log "  journal fault baseline set for this boot: $(cat "$D/seen")"
else
	log "  keeping the existing fault baseline $(cat "$D/seen" 2>/dev/null) for this boot"
fi

# The round is the soak round, not a benchmark loop, and the difference is the
# whole reason the previous hunt found nothing.
#
# Five fatal resets came out of twenty retained boots of ordinary campaign
# activity, while 255 iterations of one repeated llama-bench shape produced zero.
# Whatever provokes the fault is better represented by varied work than by hours
# of the same call. So this mirrors the round that produced the only complete
# kernel trail on record: a 2048-token prefill, a correctness gate, and an
# allocation-churn sweep, with SDMA alternated on and off round by round. That
# soak faulted during the prefill of an SDMA-enabled round.
#
# SDMA is alternated rather than left off because every earlier hunt ran with it
# disabled, which is a constant this investigation has been caught holding fixed
# before. It is not evidence that SDMA is required: faults also appeared on boots
# that predate the microcode substitution.
#
# The small model keeps a round near two minutes, which is what let the original
# soak reach 253 rounds in eight hours. Every tenth round pulls in a large model
# instead, since loads of 9 to 11 GiB exercise the KFD SVM paths that the whole
# flush family lives on.
run_round () {
	local s=$(( (iters + 1) % 2 ))            # alternate 1,0,1,0
	local E=(env HSA_ENABLE_SDMA=$s GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L)
	echo "=== round $iters sdma=$s"

	if [ $((iters % 10)) -eq 0 ]; then
		local big
		case $(( (iters / 10) % 3 )) in
		0) big=/opt/models/qwen3-8b-q8_0.gguf ;;
		1) big=/opt/models/deepseek-r1-14b.gguf ;;
		*) big=/opt/models/qwen3.6-35b-a3b-iq2m.gguf ;;
		esac
		echo "--- large model load: $big"
		"${E[@]}" timeout -k 20 900 "$HIP/llama-bench" -m "$big" -ngl 99 -fa 1 -p 512 -n 32 -r 1
		return
	fi

	echo "--- prefill"
	"${E[@]}" timeout -k 20 900 "$HIP/llama-bench" -m "$Q15" -ngl 99 -fa 1 -p 2048 -n 0 -r 3
	echo "--- gate"
	"${E[@]}" timeout -k 20 900 "$HIP/llama-perplexity" -m "$Q15" --no-mmap -ngl 99 -fa on \
		-c 4096 -f "$WIKI" --chunks 4 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+"
	echo "--- allocation churn"
	"${E[@]}" timeout -k 20 1800 "$HIP/test-backend-ops" -o MUL_MAT 2>&1 | tail -3
}

while [ ! -f "$D/STOP" ]; do
  iters=$((iters+1)); echo $iters > "$D/iters"
  t0=$(date +%s)
  run_round > "$D/run.log" 2>&1
  rc=$?
  el=$(( $(date +%s) - t0 ))
  v=$(grep -aoE "pp2048 \| +[0-9.]+|tg32 \| +[0-9.]+" "$D/run.log" | grep -oE "[0-9.]+$" | tail -1)
  g=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/run.log" | grep -oE "[0-9.]+$")
  sd=$(grep -aoE "^=== round [0-9]+ sdma=[01]" "$D/run.log" | grep -oE "[01]$")
  # A gate that moves is a correctness event even without a kernel fault, and it
  # would otherwise pass unnoticed in a hunt that only watches the journal.
  if [ -n "${g:-}" ] && [ "$g" != "$REF4" ]; then
    log "  iteration $iters: GATE DEVIATES, $g against $REF4, sdma=${sd:-?}"
    cp "$D/run.log" "$D/hits/gate$iters.log"
  fi

  # A round whose workload died is evidence too, and the first version ignored it
  # entirely: it judged a round by the journal delta alone and logged eight
  # crashing rounds as "clean" while every process in them was dumping core.
  crashed=0
  if [ $rc -ne 0 ] || grep -qaiE "segmentation fault|core dumped|Naruszenie ochrony|zrzut pami" "$D/run.log"; then
    crashed=1
    log "  iteration $iters: WORKLOAD DIED in ${el}s (rc=$rc), see hits/crash$iters.log"
    cp "$D/run.log" "$D/hits/crash$iters.log"
  fi

  now=$(faults_now)
  if [ "${now:-0}" -gt "$(cat "$D/seen" 2>/dev/null || echo 0)" ] || [ $crashed -eq 1 ]; then
    r=$(sudo journalctl -b 0 --no-pager 2>/dev/null | grep -c "GPU reset begin" || echo 0)
    log "  iteration $iters: NATURAL FAULT after ${el}s, host alive, GPU reset lines this boot=$r (rc=$rc rate=${v:-none})"
    sudo journalctl -b 0 --no-pager 2>/dev/null \
      | grep -E "$PAT|in page starting|PERMISSION_FAULTS|RW:|Faulty UTCL2|Process " \
      | tail -60 > "$D/hits/iter$iters.txt"
    echo "$now" > "$D/seen"
    # The whole point: interrogate the state now, before anything else touches
    # the GPU, rather than launching the next round and calling that a result.
    usability "iter$iters"
    log "  iteration $iters: battery done, hunting continues"
  else
    # Log every tenth round, and every round that ran a gate. The first version
    # logged only multiples of ten, which are exactly the large-model rounds that
    # skip the gate, so every line read "gate=none" and the gate could have been
    # returning nothing on every normal round without that ever being visible.
    # The deviation check itself always ran; what was missing was any evidence
    # that it had something to check.
    if [ -n "${g:-}" ]; then
      log "  iteration $iters: clean in ${el}s (rc=$rc rate=${v:-none} gate=$g sdma=${sd:-?})"
    elif [ $((iters % 10)) -eq 0 ]; then
      log "  iteration $iters: clean in ${el}s (rc=$rc rate=${v:-none} large-model round, no gate, sdma=${sd:-?})"
    fi
  fi
done
log "=== stopped by request after $iters iterations"
