#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# List ELF shared objects with LOCAL __extendhfsf2 and classify the first instruction.
for f in "$@"; do
  syms=$(readelf -Ws "$f" 2>/dev/null | awk "\$8==\"__extendhfsf2\" && \$7!=\"UND\" {print \$2}")
  [ -z "$syms" ] && continue
  a=$(echo $syms | cut -d" " -f1)
  ins=$(objdump -d --no-show-raw-insn --start-address=0x$a --stop-address=$((0x$a+12)) "$f" 2>/dev/null | grep -E "^ +[0-9a-f]+:" | sed -n 2p | cut -f2-)
  case "$ins" in *%edi*) k=BROKEN;; *xmm0*) k=ok;; *) k="?($ins)";; esac
  echo "$k $f"
done
