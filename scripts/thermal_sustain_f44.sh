#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does the board hold its clock under sustained load, and how hot does it get?
# Runs prefill continuously for N minutes while sampling the GPU clock step, edge and Tctl temperatures and
# the governor's throttle log. The 1500 MHz policy this repository measures with was chosen because the
# package default (2000 MHz at fixed 1000 mV) overheats within minutes; this checks that 1500 does not.
# Usage: thermal_sustain_f44.sh [minutes] [dir]
set -u
MIN=${1:-20}; D=${2:-~/thermal-f44}; mkdir -p "$D"
export HSA_ENABLE_SDMA=0
B=~/llama-master/build-hip-f44/bin
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }
log "config: $(tr -d ' \n' < /etc/oberon-config.yaml), throttle lines before: $(sudo journalctl -b 0 -u oberon-governor --no-pager | grep -ci throttl)"
( while :; do
    printf "%s %s %s %s\n" "$(date +%s)" \
      "$(grep '\*' /sys/class/drm/card*/device/pp_dpm_sclk | grep -oE '[0-9]+Mhz')" \
      "$(sensors 2>/dev/null | grep -oE 'edge: *\+[0-9.]+' | grep -oE '[0-9.]+')" \
      "$(sensors 2>/dev/null | grep -oE 'Tctl: *\+[0-9.]+' | grep -oE '[0-9.]+')"
    sleep 5
  done > "$D/samples.txt" ) &
SAMPLER=$!
end=$(( $(date +%s) + MIN*60 )); round=0
while [ "$(date +%s)" -lt "$end" ]; do
  round=$((round+1))
  timeout -k 20 900 $B/llama-bench -m /opt/models/qwen3-8b-q8_0.gguf -mmp 0 -ngl 99 -fa on -p 2048 -n 0 -r 3 > "$D/pp_$round.log" 2>&1
  log "round $round pp2048=$(grep -aoE 'pp2048 +\| +[0-9.]+' "$D/pp_$round.log" | grep -oE '[0-9.]+$')"
done
kill $SAMPLER 2>/dev/null
log "clock steps seen: $(awk '{print $2}' "$D/samples.txt" | sort | uniq -c | tr '\n' ' ')"
log "edge min/max: $(awk '{print $3}' "$D/samples.txt" | sort -n | sed -n '1p;$p' | tr '\n' ' ') Tctl min/max: $(awk '{print $4}' "$D/samples.txt" | sort -n | sed -n '1p;$p' | tr '\n' ' ')"
log "throttle lines after: $(sudo journalctl -b 0 -u oberon-governor --no-pager | grep -ci throttl)"
log DONE
