#!/usr/bin/env bash
# Do the two reset paths fail the same way?
#
# Every localisation of the host stall was driven through debugfs, which enters by
# amdgpu_device_gpu_recover() with AMDGPU_RESET_SRC_USER. The resets that actually
# take this board down under load enter by amdgpu_amdkfd_gpu_reset() with
# AMDGPU_RESET_SRC_HWS. That the two fail identically has been assumed since the
# bisect began and never observed. If they diverge, the whole chain from the GFX
# resume down to a single MMIO read describes the deliberate reset and not the
# one that matters.
#
# One build answers it, because the two probes touch different files: the register
# probe instruments gfx_v10_0.c, and the KFD probe adds a parameter to
# amdgpu_amdkfd.c that calls the fatal path directly. Install once, then trigger
# each way with netconsole armed and compare where the capture stops.
#
#   bash scripts/kiq_reg_probe.sh          # instruments gfx_v10_0.c
#   bash scripts/kfd_reset_probe.sh        # adds the parameter, builds, installs
#   reboot, then from the workstation:
#   scripts/reset_trial_repeat.sh path-debugfs 3 bc250_kiq_regprobe=1
#   # and the KFD arm, which needs gpu_recovery at the driver default:
#   scripts/reset_trial_repeat.sh path-kfd 3 bc250_kiq_regprobe=1 bc250_test_kfd_reset=1
#
# The KFD arm only reaches the reset with amdgpu.gpu_recovery unset or -1, since
# the guard is the whole point of that parameter; boot accordingly, and expect the
# board to die every time rather than to survive as it does under gpu_recovery=0.
#
# Read the two sets of captures against each other. Same last line means the paths
# converge and the localisation covers both. Different last lines mean the bisect
# needs redoing on the path that takes the board down.
set -eu
echo "This script documents the sequence rather than running it unattended:"
sed -n '/^#   bash/,/^#   # and the KFD arm/p' "$0" | sed 's/^# \?//'
echo
echo "Each trigger wedges the board, so run the arms one at a time and let"
echo "reset_trial_repeat.sh handle recovery."
