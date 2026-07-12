#!/bin/bash
# build_rocblas_gfx1013.sh — build a NATIVE gfx1013 rocBLAS (no gfx1010 override) on a
# Fedora-system ROCm install. This follows the intent of ROCm/rocm-libraries PR #8838
# (add gfx1013 to rocBLAS/Tensile) and adapts it to Fedora 43's system ROCm 6.4.2, where
# everything lives under /usr rather than /opt/rocm.
#
# This is offered as a worked example, not a polished installer. It worked once, on one board,
# on one day; treat every path and version as something to re-check on yours. Corrections welcome.
#
# What made it non-trivial on Fedora (each was a separate wall):
#   1. `install.sh -d` tries `dnf install python34` and aborts on F43 -> don't pass -d; deps are present.
#   2. ROCm is in /usr, not /opt/rocm -> ROCM_PATH=/usr, CXX/CC = /usr/lib64/rocm/llvm/bin/clang++|clang.
#   3. hipBLASLt is not installed -> --no_hipblaslt.
#   4. The gfx1013 target-id check fails as a false negative because clang looks for device libs
#      under $ROCM_PATH/amdgcn/bitcode -> set HIP_DEVICE_LIB_PATH to the real bitcode dir.
#   5. Tensile wants a C++ msgpack cmake package Fedora doesn't ship -> install msgpack-devel and
#      drop a tiny header-only shim config (see below).
#   6. Tensile calls the assembler `amdclang++`; Fedora ships `clang++` -> symlink + env vars.
#   7. Tensile's SupportedISA / AsmCaps and the Tensile+rocBLAS C++ enums have no gfx1013 -> add it.
#   8. rocBLAS needs roctracer/roctx.h -> install roctracer-devel.
set -u

B=/usr/lib64/rocm/llvm/bin
DEVLIB=/usr/lib64/rocm/llvm/lib/clang/19/amdgcn/bitcode   # adjust "19" to your rocm-llvm version

echo "== 0. distro deps (may need sudo; skip any already present)"
sudo dnf install -y msgpack-devel roctracer-devel 2>&1 | tail -2 || true
pip install --user -q msgpack joblib pyyaml 2>&1 | tail -1 || true

echo "== 1. header-only msgpack-cxx shim (Fedora has no msgpack-cxx-devel)"
sudo mkdir -p /usr/lib64/cmake/msgpack-cxx
sudo tee /usr/lib64/cmake/msgpack-cxx/msgpack-cxx-config.cmake >/dev/null <<'CM'
if(NOT TARGET msgpackc-cxx)
  add_library(msgpackc-cxx INTERFACE IMPORTED)
  set_target_properties(msgpackc-cxx PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "/usr/include")
endif()
set(msgpack_FOUND TRUE)
set(msgpack-cxx_FOUND TRUE)
CM

echo "== 2. amdclang++/amdclang symlinks so Tensile finds its assembler"
sudo ln -sf $B/clang++ $B/amdclang++
sudo ln -sf $B/clang   $B/amdclang

echo "== 3. clone rocBLAS docs/6.4.2 + Tensile be49885 (the version pair pinned by rocBLAS 6.4.2)"
cd ~
rm -rf ~/Tensile ~/rocBLAS
git clone -q https://github.com/ROCm/Tensile ~/Tensile
( cd ~/Tensile && git checkout -q be49885fce2a61b600ae4593f1c2d00c8b4fa11e )
git clone -q --depth 1 -b docs/6.4.2 https://github.com/ROCm/rocBLAS ~/rocBLAS

echo "== 4. add gfx1013 to rocBLAS's OWN Tensile (rocBLAS re-fetches Tensile into build/release/virtualenv;"
echo "      that copy is patched below AFTER the first configure creates it)."
echo "   -> run this script once; if the build stops at 'Unsupported target gfx1013' or the manifest"
echo "      verify, apply the gfx1013 edits to build/release/virtualenv/.../Tensile as documented in"
echo "      the repo README (SupportedISA + AsmCaps in Common.py/AsmCaps.py; Processor + LazyLoadingInit"
echo "      enums in AMDGPU.hpp / PlaceholderLibrary.hpp; and the gfx1013 enum + deviceString branches"
echo "      in rocBLAS library/src handle.hpp/handle.cpp/tensile_host.cpp), then re-run 'make -C build/release'."

echo "== 5. build"
cd ~/rocBLAS
export PATH=$B:$PATH
export ROCM_PATH=/usr
export CXX=$B/clang++ CC=$B/clang
export HIP_DEVICE_LIB_PATH=$DEVLIB
export TENSILE_ROCM_ASSEMBLER_PATH=$B/clang++
export TENSILE_ROCM_OFFLOAD_BUNDLER_PATH=$B/clang-offload-bundler
export Tensile_TEST_LOCAL_PATH=$HOME/Tensile
python3 ./rmake.py -a gfx1013 -j 4 --no_hipblaslt

L=$(ls build/release/library/src/librocblas.so* 2>/dev/null | head -1)
if [ -n "$L" ]; then
  echo "BUILT $L"
  echo "gfx1013 code objects: $(ls build/release/Tensile/library/*gfx1013* 2>/dev/null | wc -l) files"
  echo "e.g. $(file build/release/Tensile/library/Kernels.so-000-gfx1013.hsaco 2>/dev/null)"
else
  echo "no librocblas.so yet — see the gfx1013 Tensile-enablement note in step 4 and the README."
fi
