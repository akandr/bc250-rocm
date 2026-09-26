#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
#
# Build a working rocprofv3 for gfx1013 on Fedora 44. Two separate things are wrong out of the box,
# and both have to be fixed or the profiler produces an empty report with no error.
#
#   1. Fedora's rocm-runtime is built without rocprofiler-register. ROCR-Runtime's CMakeLists calls
#      find_package(rocprofiler-register) and silently disables the handshake when it is absent;
#      Fedora does not BuildRequire it, so libhsa-runtime64.so has no rocprofiler_register_*
#      symbols and no profiler can attach. This is not specific to gfx1013 or to this board: on a
#      stock Fedora ROCm install no ROCm profiler works on any GPU. Fixed here by rebuilding the
#      runtime into /opt with the package installed, leaving the system library untouched.
#
#   2. rocprofiler-sdk's counter_defs.yaml lists gfx1010, gfx1030, gfx1031 and gfx1032, and the
#      lookup in metrics.cpp matches the agent name exactly, so gfx1013 resolves to no counters at
#      all. The literal "gfx10" entry in those lists is a key like any other, not a wildcard. Fixed
#      here by adding gfx1013 beside every gfx1010. Whether the gfx1010 block layout is actually
#      right for gfx1013 is an assumption this makes and scripts/counter_validate.cpp then tests.
#
# aqlprofile, the layer that programs the counters, needs no change: it dispatches on a gfx name
# prefix ("gfx10"), so gfx1013 already selects its generic gfx10 command builder.
#
# Nothing here replaces a system package. Both builds install under /opt and are selected with
# PATH and LD_LIBRARY_PATH at run time.
set -eu

PREFIX_RP=${PREFIX_RP:-/opt/rocprof-gfx1013}
PREFIX_HSA=${PREFIX_HSA:-/opt/rocr-profreg}
TAG=${TAG:-rocm-7.1.1}          # match the ROCm the board already runs
SRC=${SRC:-$HOME}
JOBS=${JOBS:-$(nproc)}

echo "== packages"
# Deliberately not a bare 'dnf install': on this board an unconstrained transaction has upgraded
# Mesa and pruned a kernel. Inspect the plan first, and keep mesa and kernel out of it.
sudo dnf install --assumeno --setopt=installonly_limit=0 -x 'mesa*' -x 'kernel*' \
    aqlprofile-devel rocprofiler-register-devel elfutils-devel sqlite-devel || true
read -rp "install the above? [y/N] " a
[ "$a" = y ] && sudo dnf install -y --setopt=installonly_limit=0 -x 'mesa*' -x 'kernel*' \
    aqlprofile-devel rocprofiler-register-devel elfutils-devel sqlite-devel

echo "== 1. HSA runtime with the rocprofiler-register handshake"
cd "$SRC"
[ -d ROCR-Runtime ] || git clone --depth 1 -b "$TAG" https://github.com/ROCm/ROCR-Runtime.git
cd ROCR-Runtime
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="$PREFIX_HSA" \
    -DCMAKE_PREFIX_PATH="/usr/lib64/cmake;/usr/lib64/rocm/llvm/lib/cmake" \
    -DCMAKE_SHARED_LINKER_FLAGS=-ldrm_amdgpu \
    -DINCLUDE_PATH_COMPATIBILITY=OFF -DIMAGE_SUPPORT=OFF | grep -i 'rocprofiler-register' \
  || { echo "rocprofiler-register was not found; the handshake would be silently omitted"; exit 1; }
ninja -C build -j "$JOBS"
sudo ninja -C build install
nm -D --undefined-only "$PREFIX_HSA"/lib64/libhsa-runtime64.so.1.* | grep -q rocprofiler_register \
  || { echo "built runtime still has no rocprofiler_register symbols"; exit 1; }

echo "== 2. rocprofiler-sdk, with gfx1013 added to the counter definitions"
cd "$SRC"
[ -d rocprofiler-sdk ] || git clone --depth 1 -b "$TAG" https://github.com/ROCm/rocprofiler-sdk.git
cd rocprofiler-sdk
git submodule update --init --recursive --depth 1

Y=source/share/rocprofiler-sdk/counter_defs.yaml
grep -q '      - gfx1013' "$Y" || sed -i 's/^      - gfx1010$/      - gfx1010\n      - gfx1013/' "$Y"
echo "   counters now listing gfx1013: $(grep -c '      - gfx1013' "$Y")"

# Fedora folds libhsakmt into rocm-runtime and ships no hsakmt-config.cmake, which
# rocprofiler-sdk's find_package(hsakmt CONFIG REQUIRED) insists on. Supply a minimal one.
mkdir -p "$SRC/cmake-shims/lib/cmake/hsakmt"
cat > "$SRC/cmake-shims/lib/cmake/hsakmt/hsakmt-config.cmake" <<'EOF'
if(NOT TARGET hsakmt::hsakmt)
  add_library(hsakmt::hsakmt INTERFACE IMPORTED)
  set_target_properties(hsakmt::hsakmt PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "/usr/include;/usr/include/hsakmt"
    INTERFACE_LINK_LIBRARIES "/usr/lib64/libhsa-runtime64.so")
endif()
set(hsakmt_FOUND TRUE)
set(hsakmt_VERSION "1.0.6")
EOF
printf 'set(PACKAGE_VERSION "1.0.6")\nset(PACKAGE_VERSION_COMPATIBLE TRUE)\n' \
    > "$SRC/cmake-shims/lib/cmake/hsakmt/hsakmt-config-version.cmake"

# gcc 16 no longer pulls <cstdint> in transitively and the vendored elfio and yaml-cpp copies
# rely on the old behaviour. Add the include where it is missing.
for f in external/elfio/elfio/elf_types.hpp external/yaml-cpp/src/emitterutils.cpp; do
  grep -q '#include <cstdint>' "$f" || sed -i '1i #include <cstdint>' "$f"
done

cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX_RP" \
    -DROCPROFILER_BUILD_TESTS=OFF -DROCPROFILER_BUILD_SAMPLES=OFF \
    -DCMAKE_PREFIX_PATH="/usr;$SRC/cmake-shims" -DROCM_PATH=/usr
ninja -C build -j "$JOBS"
sudo ninja -C build install

cat <<EOF

Built. Use it with:

  export PATH=$PREFIX_RP/bin:\$PATH
  export LD_LIBRARY_PATH=$PREFIX_HSA/lib64:\$LD_LIBRARY_PATH
  rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU -- ./your-program

Check the counters before believing any of them:

  hipcc -O3 --offload-arch=gfx1013 counter_validate.cpp -o counter_validate -lamdhip64
  rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU -- ./counter_validate 2560

SQ_WAVES must read 2560 and SQ_INSTS_VALU must read 8192 per wave plus about eleven.
EOF
