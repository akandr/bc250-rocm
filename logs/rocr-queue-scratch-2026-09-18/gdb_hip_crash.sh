#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Run the exact September reproduction under gdb with the patched ROCr, so the crash lands in HIP, and
# take a symbolised backtrace with Fedora's debuginfo (extracted under ~/dbg, not installed).
# CPU-light apart from the model's prefill; needs the GPU free.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/rocr-repro; mkdir -p $O
DBG=$HOME/dbg/usr/lib/debug
cat > $O/gdb.cmds <<'G'
set pagination off
set confirm off
set debug-file-directory /home/akandr/dbg/usr/lib/debug:/usr/lib/debug
handle SIGSEGV stop print
run
echo \n=== faulting thread ===\n
bt 25
echo \n=== frame 0 registers ===\n
info registers rip rdi rsi rax rbx
echo \n=== locals in frames 0-3 ===\n
frame 0
info locals
frame 1
info locals
frame 2
info locals
frame 3
info locals
echo \n=== all threads (short) ===\n
thread apply all bt 6
quit
G
LD_LIBRARY_PATH=$HOME/rocr-src/lib:$L timeout -k 30 1500 gdb -q -batch -x $O/gdb.cmds --args $M/build-hip-f32mv/bin/llama-bench -m /opt/models/qwen3-14b.gguf -ngl 99 -fa on -p 0 -n 64 -d 16384 -r 2 > $O/gdb-hip-crash.log 2>&1
echo "gdb rc=$?"; grep -n "=== faulting thread" -A 30 $O/gdb-hip-crash.log | head -45
