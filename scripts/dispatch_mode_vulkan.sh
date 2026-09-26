#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Control: does the bimodality appear on Vulkan too, or is it specific to HIP without graph capture?
set -u
VKB=${VKB:-$HOME/llama-new/build-vk/bin}
N=${N:-20}
cd /tmp
echo "run  vulkan_floor  edge"
for i in $(seq 1 "$N"); do
  v=$(LD_LIBRARY_PATH=$VKB ./df_vk 2>/dev/null | grep -E "^ +1024 " | head -1 | awk '{print $2}')
  printf "%3d  %12s  %s\n" "$i" "$v" "$(sensors 2>/dev/null | awk '/edge/{print $2}')"
done
