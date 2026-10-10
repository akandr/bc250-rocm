#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# 1 Hz sampler of the SMU's own metrics (gpu_metrics v2.2) while the campaign runs: one line per minute with
# the seconds spent at each real gfxclk, the GPU and SoC temperature ranges, any throttle flag, and whether a
# llama process was running. The probe sees one instant a minute; this shows how much of each minute a run
# really spent at 1500 MHz; clock_raw.log keeps every second, so a run's measurement window can be checked.
# The packaged oberon-governor drops to its lowest OPP for 5 s whenever the GPU
# reaches 95 C or the SoC 90 C (GOV_GFX_TEMP_LIM, GOV_SOC_TEMP_LIM in its governor.hpp).
#
# While the patched governor runs (GPU limit 100 C, systemd drop-in gfx100.conf) this is also its safety net:
# any SMU throttle flag, a GPU at 101 C or more, or a new GPU fault line in the kernel log puts the packaged
# governor back at once, noted in ~/.campaign_probe/governor_reverted and in the campaign log.
echo -1000 | sudo tee /proc/$$/oom_score_adj >/dev/null 2>&1
G=/sys/class/drm/card0/device/gpu_metrics
O=~/.campaign_probe/clock1hz.log
RAW=~/.campaign_probe/clock_raw.log   # one line a second: epoch, real gfxclk MHz, GPU temperature in centi-C
D=/etc/systemd/system/oberon-governor.service.d/gfx100.conf
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
# new kernel-log lines since the last check only: a full scan of this boot's kernel log takes 3 s of a core,
# which every 30 s is load and heat the measurement does not need
CUR=~/.campaign_probe/sampler.cursor
sudo -n rm -f "$CUR"; sudo -n journalctl -b 0 -k -n 1 --no-pager -q --cursor-file="$CUR" >/dev/null 2>&1
newfaults () { sudo -n journalctl -b 0 -k --no-pager -q --cursor-file="$CUR" 2>/dev/null | grep -ciE "$PAT"; }
revert () {
  [ -f "$D" ] || return 0
  sudo -n rm -f "$D"; sudo -n systemctl daemon-reload; sudo -n systemctl restart oberon-governor
  echo "$(date '+%F %T') $1" > ~/.campaign_probe/governor_reverted
  echo "[$(date '+%F %T')] GOVERNOR: back to the packaged governor (GPU limit 95 C): $1" >> ~/ceilings-nosnap/log
}
declare -A cnt
cur=$(( $(date +%s) / 60 )); tmax=0; tmin=999; smax=0; smin=999; flags=; bench=0; i=0
while true; do
  now=$(date +%s); m=$(( now / 60 ))
  if [ "$m" -ne "$cur" ]; then
    line="$(( cur * 60 ))"; for k in $(printf '%s\n' "${!cnt[@]}" | sort -n); do line+=" ${k}MHz=${cnt[$k]}"; done
    echo "$line tgfx=${tmin}-${tmax} tsoc=${smin}-${smax} flags=${flags:-0} bench=$bench" >> "$O"
    cnt=(); tmax=0; tmin=999; smax=0; smin=999; flags=; bench=0; cur=$m
  fi
  ag=$(od -An -tu2 -j64 -N2 "$G" | tr -d ' ')
  tc=$(od -An -tu2 -j4 -N2 "$G" | tr -d ' '); t=$(( tc / 100 ))
  echo "$now $ag $tc" >> "$RAW"
  s=$(( $(od -An -tu2 -j6 -N2 "$G" | tr -d ' ') / 100 ))
  th=$(od -An -tx4 -j108 -N4 "$G" | tr -d ' ')
  cnt[$ag]=$(( ${cnt[$ag]:-0} + 1 ))
  (( t > tmax )) && tmax=$t; (( t < tmin )) && tmin=$t
  (( s > smax )) && smax=$s; (( s < smin )) && smin=$s
  if [ "$th" != 00000000 ]; then flags=$th; revert "SMU throttle flags 0x$th at GPU $t C, SoC $s C"; fi
  (( t >= 101 )) && revert "GPU $t C"
  if (( i % 10 == 0 )); then { pgrep -x llama-bench || pgrep -x llama-cli || pgrep -x llama-perplexit; } >/dev/null && bench=1; fi
  if (( i % 30 == 0 )) && [ -f "$D" ]; then f=$(newfaults); (( f > 0 )) && revert "$f new GPU fault line(s) in the kernel log"; fi
  i=$(( i + 1 )); sleep 1
done
