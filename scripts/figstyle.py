# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Colours and matplotlib settings shared by every figure script in this directory.

ROCm is red and Vulkan blue wherever both appear; lighter shades of each are the older or secondary
series. Figures that compare something other than the two backends (two ROCm builds, two kernels, two
precisions) use amber, teal, violet and grey, so that blue never stands for anything but Vulkan.
Backgrounds stay white so the figures read the same on GitHub's dark theme.
"""
import matplotlib.pyplot as plt

ROCM, ROCM_MID, ROCM_PALE = "#DC2626", "#F87171", "#FECACA"
ROCM_DARK = "#991B1B"
VULKAN, VULKAN_MID, VULKAN_PALE = "#2563EB", "#60A5FA", "#BFDBFE"
VULKAN_DARK = "#1E3A8A"
AMBER, AMBER_PALE = "#D97706", "#FDE68A"
TEAL, TEAL_PALE = "#0D9488", "#99F6E4"
VIOLET = "#7C3AED"
INK = "#1F2937"        # text and reference lines
GREY = "#6B7280"       # secondary text, other series
GREY_MID = "#9CA3AF"
GREY_PALE = "#D1D5DB"
NOTE = "#4B5563"       # captions under a figure
GRID = "#E5E7EB"
# light backgrounds behind "ROCm faster" / "Vulkan faster" regions of a ratio axis
ROCM_BAND, VULKAN_BAND = "#FEF2F2", "#EFF6FF"


def apply():
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.dpi": 150,
        "axes.edgecolor": GREY_MID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "normal",
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False,
    })
