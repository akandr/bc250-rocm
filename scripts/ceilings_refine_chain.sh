#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Phase D of scripts/ceilings_nosnap_chain.sh. Phase C bisected the 1.5B's q8_0 and q4_0 cells only to half of the
# bracket the doubling found, which leaves the arms' brackets overlapping (q8_0: hip 589824-786432, hipng 638976-851968,
# vk 688128-917504), so they cannot be told apart. Once the chain marks ALL-DONE, this runs the same harness on the same
# directory with RESDIV=4: the harness takes the resolution from the bracket it finds, which is now the halved one, so a
# quarter of it is an eighth of the doubling bracket, the resolution of phases A and B. Every known result is reused and
# only the missing midpoints run, the q8_0 cells first because they are the shorter runs.
set -u
D=~/ceilings-nosnap
export LLAMA_BENCH_NO_DEPTH_CACHE=1 HIP=~/llama-bench-nosnap/hip VK=~/llama-bench-nosnap/vk   # as the chain does
mark () { echo "[$(date '+%F %T')] $*" >> "$D/log"; sync; }
# wait for the chain's own end marker, a line of its own
until grep -qE '^\[[0-9: -]+\] ALL-DONE$' "$D/log"; do sleep 60; done
mark "chain: phase D, refining the 1.5B q8_0/q4_0 cells to an eighth of their doubling bracket (RESDIV=4 on the halved brackets)"
SEED=$D/seed-ok.tsv MODELS=qwen2.5-1.5b-q4km KVS="q8_0 q4_0" RESDIV=4 TMO=604800 bash ~/ceilings_bisect.sh "$D" ||
  { mark "chain stopped in phase D"; exit 1; }
mark REFINE-DONE
