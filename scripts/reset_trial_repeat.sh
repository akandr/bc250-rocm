#!/usr/bin/env bash
# Trigger a GPU reset N times and keep every capture, one file per trial.
#
# Runs on the workstation, not the board: it needs the netconsole listener and
# the power switch, and the board stops answering partway through each trial.
#
# Repetition is the point. Single captures on this board have produced three
# effects that looked convincing and reversed on resampling, and the reset work
# added a fourth: one run in which extra prints let the KIQ resume get further
# was read as evidence that prints help, when one success out of one attempt is
# one sample. Anything claimed about this stall needs a rate, not an anecdote.
#
# Netconsole is UDP with no retransmission, so a lost packet and a stall look
# identical. The listener is verified with a /dev/kmsg marker before every
# trigger, by retry rather than by aborting on the first miss, since a listener
# started right after modprobe can miss the first packets.
#
# Usage: reset_trial_repeat.sh <label> <count> [param=value ...]
#   scripts/reset_trial_repeat.sh kiq-fine 5
#   scripts/reset_trial_repeat.sh gap 3 bc250_kiq_pre_mode=3 bc250_kiq_mid_mode=0
#   TRIGGER=kfd scripts/reset_trial_repeat.sh path-kfd 3 bc250_kiq_regprobe=1
set -u
LABEL=${1:?usage: reset_trial_repeat.sh <label> <count> [param=value ...]}
COUNT=${2:?usage: reset_trial_repeat.sh <label> <count> [param=value ...]}
shift 2
BOARD=${BOARD:-bc250}
# One runner at a time. Two of these were once started against the same board:
# the first was still waiting out its recovery while the second began triggering,
# and they fought over the board and the UDP listener, producing two skipped
# trials that looked like netconsole trouble rather than like a self-inflicted
# collision.
LOCK=${LOCK:-/tmp/bc250_reset_trial.lock}
if ! mkdir "$LOCK" 2>/dev/null; then
	echo "another reset trial runner holds $LOCK; refusing to start" >&2
	exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
PORT=${PORT:-6969}
LISTENER_IP=${LISTENER_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')}
[ -n "$LISTENER_IP" ] || { echo "cannot determine this machine's address; set LISTENER_IP" >&2; exit 1; }
OUTDIR=${OUTDIR:-logs/reset-$LABEL-$(date +%Y-%m-%d)}
mkdir -p "$OUTDIR"

# Continue numbering from whatever is already there rather than restarting at 1.
# Restarting silently overwrote a capture that had already been read and reasoned
# from, and it had not been committed yet, so the artifact behind a stated finding
# was simply gone.
FIRST=1
while [ -e "$OUTDIR/${LABEL}_rep${FIRST}.log" ]; do FIRST=$((FIRST + 1)); done
LAST=$((FIRST + COUNT - 1))
echo "$LABEL: writing reps $FIRST to $LAST in $OUTDIR"

for rep in $(seq "$FIRST" "$LAST"); do
	out="$OUTDIR/${LABEL}_rep${rep}.log"
	# The workstation's address is DHCP-assigned and has moved at least once, so
	# pass the current one rather than relying on the script's default.
	ssh -o ConnectTimeout=25 "$BOARD" "LISTENER_IP=$LISTENER_IP bash ~/netconsole_capture.sh >/dev/null 2>&1" </dev/null
	pkill -f "nc -u -l $PORT" 2>/dev/null; sleep 1
	rm -f "$out"; (nc -u -l "$PORT" > "$out" 2>/dev/null &); sleep 2

	ok=0
	for t in 1 2 3 4 5 6; do
		ssh -o ConnectTimeout=25 "$BOARD" \
			"echo MARK-$LABEL-$rep-$t | sudo tee /dev/kmsg >/dev/null" </dev/null >/dev/null 2>&1
		sleep 3
		grep -q "MARK-$LABEL-$rep-$t" "$out" 2>/dev/null && { ok=1; break; }
	done
	if [ $ok -ne 1 ]; then
		pkill -f "nc -u -l $PORT" 2>/dev/null
		rm -f "$out"
		# An unverified listener usually means the board is not up, so recover
		# rather than burning the remaining trials against a dead machine.
		if ssh -o ConnectTimeout=10 "$BOARD" 'echo up' </dev/null 2>/dev/null | grep -q up; then
			echo "TRIAL $LABEL rep$rep: board up but netconsole unverified, skipped"
		else
			echo "TRIAL $LABEL rep$rep: board down, power cycling and retrying"
			bash "$(dirname "$0")/bc250_power" off >/dev/null 2>&1
			sleep 15
			bash "$(dirname "$0")/bc250_power" on >/dev/null 2>&1
			for i in $(seq 1 20); do
				ssh -o ConnectTimeout=10 "$BOARD" 'echo up' </dev/null 2>/dev/null | grep -q up && break
				sleep 15
			done
		fi
		continue
	fi

	for kv in "$@"; do
		ssh -o ConnectTimeout=25 "$BOARD" \
			"echo ${kv#*=} | sudo tee /sys/module/amdgpu/parameters/${kv%%=*} >/dev/null" \
			</dev/null >/dev/null 2>&1
	done

	# Two ways in, and which one is used is the whole point of the path
	# comparison. debugfs enters by amdgpu_device_gpu_recover() with
	# AMDGPU_RESET_SRC_USER; the probe parameter enters by
	# amdgpu_amdkfd_gpu_reset() with AMDGPU_RESET_SRC_HWS, which is what the
	# resets that actually take this board down under load use. The KFD arm only
	# reaches a reset with amdgpu.gpu_recovery at the driver default, since the
	# guard is what that parameter turns off.
	#
	# For debugfs, reading the node is what triggers the reset; writing to it does
	# nothing. The path is discovered because debugfs exposes the device under
	# both a minor number and its PCI address. Either way the trigger runs in the
	# foreground of its ssh, because a command backgrounded on the far side is
	# killed by the hangup before it runs.
	case ${TRIGGER:-debugfs} in
	kfd)
		ssh -o ConnectTimeout=25 "$BOARD" \
			'echo 1 | sudo tee /sys/module/amdgpu/parameters/bc250_test_kfd_reset' \
			</dev/null >/dev/null 2>&1 &
		;;
	*)
		ssh -o ConnectTimeout=25 "$BOARD" \
			'N=$(sudo find /sys/kernel/debug/dri -maxdepth 2 -name amdgpu_gpu_recover | head -1); sudo cat "$N"' \
			</dev/null >/dev/null 2>&1 &
		;;
	esac
	trig=$!
	sleep 75
	kill $trig 2>/dev/null
	pkill -f "nc -u -l $PORT" 2>/dev/null
	echo "TRIAL $LABEL rep$rep: $(grep -ac . "$out") lines captured"

	# The board no longer comes back on its own from this wedge, so waiting the
	# full hardware-watchdog timeout costs ten minutes and buys nothing.
	back=0
	for i in $(seq 1 9); do
		ssh -o ConnectTimeout=10 "$BOARD" 'echo up' </dev/null 2>/dev/null | grep -q up && { back=1; break; }
		sleep 20
	done
	if [ $back -ne 1 ]; then
		echo "TRIAL $LABEL rep$rep: recovering by power cycle"
		bash "$(dirname "$0")/bc250_power" off >/dev/null 2>&1
		sleep 15
		bash "$(dirname "$0")/bc250_power" on >/dev/null 2>&1
		for i in $(seq 1 20); do
			ssh -o ConnectTimeout=10 "$BOARD" 'echo up' </dev/null 2>/dev/null | grep -q up && break
			sleep 15
		done
	fi
done
echo "$LABEL: $COUNT trials done, captures in $OUTDIR"
