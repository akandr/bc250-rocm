#!/bin/bash
# sweep_cfg.sh LABEL MODE FREQ MV
#   MODE=oberon : leave oberon-governor running (dynamic 1000-1500 MHz)
#   MODE=fixed  : stop oberon, set fixed FREQ (MHz) / MV (mV) via pp_od_clk_voltage
# Runs the compute_probe sweep and logs everything to ~/sweeplogs/$LABEL/
set -u
LABEL="$1"; MODE="$2"; FREQ="${3:-0}"; MV="${4:-0}"
D=~/sweeplogs/"$LABEL"; mkdir -p "$D"
exec > >(tee "$D/sweep.log") 2>&1
SYS=/sys/class/drm/card1/device
PMINFO=/sys/kernel/debug/dri/1/amdgpu_pm_info

echo "=== sweep $LABEL mode=$MODE freq=$FREQ mv=$MV $(date -Is) ==="
uname -r; cat /proc/cmdline
cat /sys/module/amdgpu/parameters/bc250_cc_write_mode 2>/dev/null
journalctl -k -b 0 --no-pager | grep -m1 active_cu_number

# stop GPU consumers (pre-flight rule)
sudo systemctl stop ollama queue-runner 2>/dev/null
sleep 3

if [ "$MODE" = fixed ]; then
  sudo systemctl stop oberon-governor
  sleep 2
  echo "vc 0 $FREQ $MV" | sudo tee $SYS/pp_od_clk_voltage >/dev/null
  echo "c" | sudo tee $SYS/pp_od_clk_voltage >/dev/null
  sleep 2
else
  sudo systemctl start oberon-governor 2>/dev/null
fi
echo "--- clock state after setup:"
cat $SYS/pp_od_clk_voltage
sudo grep -E "SCLK|VDDGFX|Temperature" $PMINFO

temp() { sudo grep "GPU Temperature" $PMINFO | grep -oE "[0-9]+"; }
cool() { # wait until GPU <= 82 C (max 10 min)
  for i in $(seq 1 60); do t=$(temp); [ "${t:-0}" -le 82 ] && return; sleep 10; done
  echo "WARN: still hot after 10 min: $(temp) C"
}

# telemetry sampler
( while true; do
    echo "$(date +%s) $(sudo grep -E 'SCLK|VDDGFX|Temperature' $PMINFO | tr '\n' ' ')"
    sleep 2
  done ) > "$D/telemetry.log" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

dmesg_mark() { sudo dmesg | tail -1 | awk -F'[][]' '{print $2}'; }
dmesg_delta() { sudo dmesg | awk -v m="$1" -F'[][]' '$2>m' | tail -30; }

run_one() { # run_one nblocks inner tag
  local nb=$1 inner=$2 tag=$3
  cool
  local m=$(dmesg_mark)
  echo "--- RUN $tag nblocks=$nb inner=$inner temp=$(temp)C $(date -Is)"
  HSA_ENABLE_SDMA=0 timeout -k 10 240 ~/compute_probe_fresh "$nb" "$inner" 1
  local rc=$?
  echo "--- RC=$rc (124=timeout/hang)"
  dmesg_delta "$m" > "$D/dmesg_$tag.log"
  [ -s "$D/dmesg_$tag.log" ] && echo "--- dmesg had output, see dmesg_$tag.log"
  return $rc
}

echo "=== sanity small kernel ==="
run_one 4096 1000 sanity

HUNG=0
for size in 16384 32768 65536; do
  for rep in 1 2 3 4; do
    run_one $size 6000 "${size}_r${rep}" || { [ $? -eq 124 ] && HUNG=$((HUNG+1)); }
    [ $HUNG -ge 3 ] && { echo "=== 3 hangs, aborting sweep (degraded)"; break 2; }
  done
done

echo "=== post-sweep small kernel (degradation check) ==="
run_one 4096 1000 post_sanity

echo "=== sweep $LABEL done $(date -Is), hangs=$HUNG ==="
