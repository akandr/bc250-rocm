#!/usr/bin/env bash
# Build PyTorch 2.9.1 from source with gfx1013 code objects.
# Kept narrow on purpose: the point is architecture coverage for the kernels
# torch compiles itself, not a full-featured wheel. Distributed, tests and the
# extra backends are all off to keep the build inside this board's 14 GB.
set -u
LOG=~/torch-build/build.log
mkdir -p ~/torch-build
cd ~/pytorch || exit 1
source ~/torchvenv/bin/activate

# ROCm here is the Fedora system packaging under /usr, not the AMD installer
# layout at /opt/rocm, so every path has to be pointed at it explicitly or
# cmake silently configures a CPU-only build.
export ROCM_PATH=/usr
export ROCM_HOME=/usr
export HIP_PATH=/usr
export ROCM_SOURCE_DIR=/usr
# The HIP build derives its compiler as ${ROCM_PATH}/llvm/bin, which does not
# exist on Fedora; the ROCm clang lives under lib64/rocm.
export HIP_CLANG_PATH=/usr/lib64/rocm/llvm/bin
export CMAKE_PREFIX_PATH=/usr/lib64/cmake:${CMAKE_PREFIX_PATH:-}
export USE_NCCL=0
# Composable Kernel is not packaged on Fedora, and it has no gfx1013 support.
export USE_ROCM_CK_GEMM=0
export USE_ROCM_CK_SDPA=0
export USE_RCCL=0
export PYTORCH_ROCM_ARCH=gfx1013
export USE_ROCM=1
export USE_CUDA=0
export USE_DISTRIBUTED=0
export USE_MKLDNN=0
export USE_FBGEMM=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export BUILD_TEST=0
export USE_KINETO=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export MAX_JOBS=6
export CMAKE_BUILD_PARALLEL_LEVEL=6
export BUILD_CAFFE2=0

{
  echo "=== build start $(date '+%F %T')"
  echo "=== arch=$PYTORCH_ROCM_ARCH jobs=$MAX_JOBS"
  python -c "import sys; print('python', sys.version)"
  echo "=== hipify"
  python tools/amd_build/build_amd.py 2>&1 | tail -5
  echo "=== bdist_wheel"
  time python setup.py bdist_wheel 2>&1
  echo "=== build end $(date '+%F %T') rc=$?"
  ls -la dist/ 2>/dev/null
} >> "$LOG" 2>&1

touch ~/torch-build/DONE
