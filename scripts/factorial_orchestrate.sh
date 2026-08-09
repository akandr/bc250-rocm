#!/usr/bin/env bash
# factorial_orchestrate.sh - drives the 2x2 factorial from the Mac.
# Cells: A=(flush0,policy0) B=(flush0,policy2) C=(flush1,policy0) D=(flush1,policy2)
# Boot order mirror-balanced: A B C D D C B A  (2 boots per cell).
# Handles boot hangs and post-test freezes via the HA power switch.
set -u
K=/boot/vmlinuz-7.1.5-100.fc43.x86_64
RES=/Users/akandr/projects/bc250_rocm/validation-2026-08/factorial_results.tsv
HAIP=http://192.168.3.159:8123
source ~/.config/bc250_reset.env   # HASS_TOKEN, HASS_SWITCH
[ -f "$RES" ] || echo -e "seq\tcell\tflush\tpolicy\tboot_outcome\tverdict" > "$RES"

ha () { curl -s -m 10 -X POST -H "Authorization: Bearer $HASS_TOKEN" \
        -H "Content-Type: application/json" -d "{\"entity_id\": \"$HASS_SWITCH\"}" \
        "$HAIP/api/services/switch/turn_$1" > /dev/null; }
power_cycle () { echo "  [power-cycle]"; ha off; sleep 4; ha on; }

up () { ssh -o ConnectTimeout=6 -o BatchMode=yes bc250 'cut -d. -f1 /proc/uptime' 2>/dev/null; }

wait_fresh () { # wait for a boot fresher than 300s; retn 0 ok, 1 gave up
  local waited=0
  while [ $waited -lt 900 ]; do
    u=$(up); if [ -n "${u:-}" ] && [ "$u" -lt 300 ]; then return 0; fi
    sleep 20; waited=$((waited+20))
  done
  return 1
}

configure () { # $1=flush(0|1) $2=policy(0|2)
  local fl=$1 po=$2
  ssh bc250 "sudo grubby --update-kernel=$K --remove-args='amdgpu.bc250_flush_pasid_kiq amdgpu.sched_policy' >/dev/null 2>&1;
             sudo grubby --update-kernel=$K --args='amdgpu.bc250_flush_pasid_kiq=$fl' >/dev/null 2>&1;
             $( [ "$po" = 2 ] && echo "sudo grubby --update-kernel=$K --args='amdgpu.sched_policy=2' >/dev/null 2>&1;" )
             sync; sleep 1; sync"
}

seq=0
for cell in A B C D D C B A; do
  seq=$((seq+1))
  case $cell in
    A) fl=0; po=0;; B) fl=0; po=2;; C) fl=1; po=0;; D) fl=1; po=2;;
  esac
  echo "=== boot $seq cell $cell (flush=$fl policy=$po) $(date +%H:%M:%S) ==="
  if ! configure $fl $po; then
    echo "  configure failed (board unreachable?); power cycling and retrying"
    power_cycle; wait_fresh || true; configure $fl $po || { echo "  give up seq $seq"; continue; }
  fi
  ssh bc250 'nohup sh -c "sleep 3; sudo -n /sbin/reboot" >/dev/null 2>&1 </dev/null & exit' 2>/dev/null
  sleep 30
  boot=OK
  if ! wait_fresh; then
    echo "  boot hang; power cycling"
    boot=BOOT_HANG_POWERCYCLED
    power_cycle
    wait_fresh || { echo -e "$seq\t$cell\t$fl\t$po\tBOOT_FAILED\t-" >> "$RES"; continue; }
  fi
  # run the battery with an overall cap; battery itself has per-step timeouts
  V=$(ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=12 -o BatchMode=yes bc250 "~/factorial_battery.sh $cell $seq 2>/dev/null | grep ^VERDICT" 2>/dev/null || true)
  if [ -z "$V" ]; then
    # board died mid-battery: that IS a result (freeze/crash)
    V="VERDICT cell=$cell idx=$seq BOARD_LOST_MID_BATTERY"
    boot="$boot+LOST"
    power_cycle
  fi
  echo "  $V"
  echo -e "$seq\t$cell\t$fl\t$po\t$boot\t$V" >> "$RES"
done
echo "=== factorial complete ==="
column -t -s $'\t' "$RES"
