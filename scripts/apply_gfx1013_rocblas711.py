#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Add gfx1013 (Cyan Skillfish, BC-250) to rocBLAS and Tensile in a ROCm/rocm-libraries checkout, so rocBLAS
# can be built with GPU_TARGETS=gfx1013. Written against tag rocm-7.1.1 (commit f322e9ab61) and applied
# unchanged to rocm-7.2.2 (dabb6df2b9): every anchor still matched.
#
# The same places as ROCm/rocm-libraries PR #8838 and Fedora's Tensile gfx1036/gfx1153 patches:
#   Tensile  Common.py            SupportedISA (10,1,3), architectureMap gfx1013
#   Tensile  AsmCaps.py           (10,1,3) capabilities, copied from (10,1,0)
#   Tensile  TensileSupportedArchitectures.cmake
#   Tensile  AMDGPU.hpp           Processor enum, toString, toProcessorId
#   Tensile  PlaceholderLibrary.hpp   LazyLoadingInit enum and file pattern
#   Tensile  Serialization/Predicates.hpp
#   rocBLAS  handle.hpp, handle.cpp, tensile_host.cpp
# Every edit is anchored on the existing gfx1012 line and must match exactly once.
#
# Usage: apply_gfx1013_rocblas711.py <rocm-libraries checkout>
import os, re, sys

root = sys.argv[1]
T = os.path.join(root, "shared/tensile/Tensile")
P = os.path.join(root, "projects/rocblas/library/src")

def edit(path, old, new):
    p = os.path.join(path)
    s = open(p).read()
    if new in s:
        print("already", os.path.relpath(p, root)); return
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{p}: anchor found {n} times:\n{old}")
    open(p, "w").write(s.replace(old, new, 1))
    print("patched", os.path.relpath(p, root))

edit(f"{T}/Common.py", "(10,1,0), (10,1,1), (10,1,2), (10,3,0)", "(10,1,0), (10,1,1), (10,1,2), (10,1,3), (10,3,0)")
edit(f"{T}/Common.py", "'gfx1012':'navi14',", "'gfx1012':'navi14', 'gfx1013':'gfx1013',")

s = open(f"{T}/AsmCaps.py").read()
if "(10, 1, 3)" not in s:
    m = re.search(r"(     \(10, 1, 0\): \{.*?\},\n)", s, re.S)
    assert m, "AsmCaps (10, 1, 0) block"
    block = m.group(1)
    anchor = "     (10, 1, 2): {"
    assert s.count(anchor) == 1, "AsmCaps (10, 1, 2) anchor"
    s = s.replace(anchor, block.replace("(10, 1, 0)", "(10, 1, 3)") + anchor, 1)
    open(f"{T}/AsmCaps.py", "w").write(s)
    print("patched shared/tensile/Tensile/AsmCaps.py")

edit(f"{T}/Source/cmake/TensileSupportedArchitectures.cmake", '        "gfx1012"\n', '        "gfx1012"\n        "gfx1013"\n')

H = f"{T}/Source/lib/include/Tensile"
edit(f"{H}/AMDGPU.hpp", "            gfx1012 = 1012,\n", "            gfx1012 = 1012,\n            gfx1013 = 1013,\n")
edit(f"{H}/AMDGPU.hpp",
     '            case AMDGPU::Processor::gfx1012:\n                return "gfx1012";\n',
     '            case AMDGPU::Processor::gfx1012:\n                return "gfx1012";\n'
     '            case AMDGPU::Processor::gfx1013:\n                return "gfx1013";\n')
edit(f"{H}/AMDGPU.hpp",
     '            else if(deviceString.find("gfx1012") != std::string::npos)\n            {\n                return AMDGPU::Processor::gfx1012;\n            }\n',
     '            else if(deviceString.find("gfx1012") != std::string::npos)\n            {\n                return AMDGPU::Processor::gfx1012;\n            }\n'
     '            else if(deviceString.find("gfx1013") != std::string::npos)\n            {\n                return AMDGPU::Processor::gfx1013;\n            }\n')
edit(f"{H}/PlaceholderLibrary.hpp", "        gfx1012,\n", "        gfx1012,\n        gfx1013,\n")
edit(f"{H}/PlaceholderLibrary.hpp",
     '        case LazyLoadingInit::gfx1012:\n            return "TensileLibrary_*_gfx1012";\n',
     '        case LazyLoadingInit::gfx1012:\n            return "TensileLibrary_*_gfx1012";\n'
     '        case LazyLoadingInit::gfx1013:\n            return "TensileLibrary_*_gfx1013";\n')
edit(f"{H}/Serialization/Predicates.hpp",
     '                iot::enumCase(io, value, "gfx1012", AMDGPU::Processor::gfx1012);\n',
     '                iot::enumCase(io, value, "gfx1012", AMDGPU::Processor::gfx1012);\n'
     '                iot::enumCase(io, value, "gfx1013", AMDGPU::Processor::gfx1013);\n')

edit(f"{P}/include/handle.hpp", "    gfx1012 = 1012,\n", "    gfx1012 = 1012,\n    gfx1013 = 1013,\n")
edit(f"{P}/handle.cpp",
     '    else if(deviceString.find("gfx1012") != std::string::npos)\n    {\n        return Processor::gfx1012;\n    }\n',
     '    else if(deviceString.find("gfx1012") != std::string::npos)\n    {\n        return Processor::gfx1012;\n    }\n'
     '    else if(deviceString.find("gfx1013") != std::string::npos)\n    {\n        return Processor::gfx1013;\n    }\n')
edit(f"{P}/tensile_host.cpp",
     '        else if(deviceString.find("gfx1012") != std::string::npos)\n        {\n            return Tensile::LazyLoadingInit::gfx1012;\n        }\n',
     '        else if(deviceString.find("gfx1012") != std::string::npos)\n        {\n            return Tensile::LazyLoadingInit::gfx1012;\n        }\n'
     '        else if(deviceString.find("gfx1013") != std::string::npos)\n        {\n            return Tensile::LazyLoadingInit::gfx1013;\n        }\n')
print("done")
