#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Configure, build and install rocBLAS 7.1.1 for gfx1013 only, on Fedora 44 (ROCm 7.1.1 packages).
#
# Expects a rocm-libraries checkout at tag rocm-7.1.1 with scripts/apply_gfx1013_rocblas711.py applied:
#   W/rocm-libraries/projects/rocblas and W/rocm-libraries/shared/tensile
# Output: W/install (lib64/librocblas.so.5*, lib64/rocblas/library). About 50 minutes on the BC-250.
#
# Usage: build_rocblas711_gfx1013.sh [W]     (W defaults to ~/rb711)
#
# Environment problems met on Fedora 44, handled or checked here:
#   - Tensile asks CMake for a msgpack-cxx target; Fedora's msgpack-devel (3.1.0) ships the headers but
#     no such config. A header-only config is generated in the build tree and passed as msgpack-cxx_DIR.
#   - Tensile runs its parallel steps through joblib, which falls back to in-process execution without
#     /dev/shm; on that path Tensile empties its own global parameters (KeyError: 'PrintIndexAssignments').
#   - Tensile requires rocm_agent_enumerator (package rocminfo) and rocBLAS requires rocm-cmake.
set -u
W=${1:-$HOME/rb711}
B=/usr/lib64/rocm/llvm/bin
S=$W/rocm-libraries/projects/rocblas
O=$W/build
[ -d "$S" ] || { echo "no rocBLAS source at $S"; exit 1; }
mountpoint -q /dev/shm || [ -w /dev/shm ] || { echo "/dev/shm is missing: Tensile would fail, mount it first"; exit 1; }
command -v rocm_agent_enumerator >/dev/null || [ -x /usr/bin/rocm_agent_enumerator ] || { echo "install rocminfo"; exit 1; }
export PATH=$B:/usr/bin:/usr/sbin ROCM_PATH=/usr
export TENSILE_ROCM_ASSEMBLER_PATH=$B/amdclang++ TENSILE_ROCM_OFFLOAD_BUNDLER_PATH=$B/clang-offload-bundler

MP=$O/msgpack-shim; mkdir -p $MP
cat > $MP/msgpack-cxx-config.cmake <<'CM'
# Header-only msgpack C++ config for Fedora's msgpack-devel, defining the target names Tensile looks for.
foreach(t msgpack-cxx msgpackc-cxx msgpackc)
  if(NOT TARGET ${t})
    add_library(${t} INTERFACE IMPORTED)
    set_target_properties(${t} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "/usr/include")
  endif()
endforeach()
set(msgpack-cxx_FOUND TRUE)
CM

cmake -S $S -B $O -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$B/amdclang -DCMAKE_CXX_COMPILER=$B/amdclang++ \
  -DCMAKE_INSTALL_PREFIX=$W/install \
  -Dmsgpack-cxx_DIR=$MP \
  -DHIP_PLATFORM=amd -DGPU_TARGETS=gfx1013 \
  -DBUILD_WITH_TENSILE=ON -DBUILD_WITH_HIPBLASLT=OFF \
  -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCHMARKS=OFF -DBUILD_FORTRAN_CLIENTS=OFF \
  -DTensile_LIBRARY_FORMAT=msgpack -DTensile_CPU_THREADS=4 \
  -DROCM_SYMLINK_LIBS=OFF || exit 1
cmake --build $O -j 4 && cmake --install $O
