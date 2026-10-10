#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Add the bc250_flush_pasid_kiq module parameter to gmc_v10_0.c of a kernel tree.

The line itself is anrp's workaround from ROCm/ROCm#6313: flush_pasid_uses_kiq = false, so PASID TLB
flushes go through MMIO instead of the KIQ ring, which on gfx1013 flushes nothing. This puts it behind a
module parameter, default 1 (stock behaviour), so that amdgpu.bc250_flush_pasid_kiq=0 on the kernel
command line turns it on and the stock behaviour stays one boot argument away. Idempotent. This is the
step scripts/ladder_prep_rung.sh carried inline; patches/amdgpu-flush-pasid-mmio.patch is the same change
hardcoded, without the parameter.

Usage: apply_flush_pasid_param.py <kernel tree>
"""
import os
import sys

if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
    sys.exit("usage: apply_flush_pasid_param.py <kernel tree>")
p = os.path.join(sys.argv[1], "drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c")
if not os.path.exists(p):
    sys.exit("not a kernel tree: %s has no drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c" % sys.argv[1])
s = open(p).read()
if "bc250_flush_pasid_kiq" in s:
    print("already applied")
    sys.exit(0)
anchor = "adev->gmc.flush_pasid_uses_kiq = !amdgpu_emu_mode;"
if anchor not in s:
    sys.exit("anchor not found in gmc_v10_0.c: %r\nport by hand" % anchor)
decl = ("static int bc250_flush_pasid_kiq = 1;\n"
        "module_param(bc250_flush_pasid_kiq, int, 0444);\n"
        "MODULE_PARM_DESC(bc250_flush_pasid_kiq, \"BC-250: 0=flush_pasid_uses_kiq=false (ROCm#6313)\");\n\n")
first_func = s.index("static ")
s = s[:first_func] + decl + s[first_func:]
s = s.replace(anchor, "adev->gmc.flush_pasid_uses_kiq = bc250_flush_pasid_kiq ? !amdgpu_emu_mode : false; /* BC-250 */", 1)
open(p, "w").write(s)
print("applied")
