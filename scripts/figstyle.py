# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Colours and matplotlib settings shared by every figure script in this directory.

ROCm is red and Vulkan blue wherever both appear; paler shades of each are the older or secondary
series. Figures that compare something other than the two backends (two ROCm builds, two kernels, two
precisions) use amber, teal, violet and grey, so that blue never stands for anything but Vulkan.

Every figure is drawn twice: BC250_THEME=light (the default) writes figures/<name>.png on white, and
BC250_THEME=dark writes figures/dark/<name>.png on GitHub's dark background. The READMEs pick one with
<picture> and prefers-color-scheme. In the dark theme the "pale" shades are dim fills and the "dark"
shades are the brighter emphasis colour, so each name keeps its role against its background.
"""
import os
import matplotlib.pyplot as plt

THEME = os.environ.get("BC250_THEME", "light")
DARK = THEME == "dark"

if not DARK:
    BG = "#FFFFFF"
    ROCM, ROCM_MID, ROCM_PALE, ROCM_DARK = "#DC2626", "#F87171", "#FECACA", "#991B1B"
    VULKAN, VULKAN_MID, VULKAN_PALE, VULKAN_DARK = "#2563EB", "#60A5FA", "#BFDBFE", "#1E3A8A"
    AMBER, AMBER_PALE = "#D97706", "#FDE68A"
    TEAL, TEAL_PALE = "#0D9488", "#99F6E4"
    GREEN = "#2A9D8F"
    VIOLET = "#7C3AED"
    PURPLE, BROWN, ORANGE = "#6A3D9A", "#B15928", "#E6550D"
    INK = "#1F2937"        # text and reference lines
    INK_SOFT = "#374151"   # value labels
    GREY = "#6B7280"       # secondary text, other series
    GREY_MID = "#9CA3AF"
    GREY_PALE = "#D1D5DB"
    NOTE = "#4B5563"       # captions under a figure
    GRID = "#E5E7EB"
    ROCM_BAND, VULKAN_BAND = "#FEF2F2", "#EFF6FF"
    BAND_ALPHA = 0.06
    # six steps from the first patch set to the last, light to strong
    REDS = ["#FECACA", "#FCA5A5", "#F87171", "#EF4444", "#DC2626", "#991B1B"]
    ON_BAR = "#FFFFFF"     # text printed inside a strong bar
else:
    BG = "#0D1117"         # GitHub dark canvas
    ROCM, ROCM_MID, ROCM_PALE, ROCM_DARK = "#F85149", "#FF7B72", "#5C2226", "#FFA198"
    VULKAN, VULKAN_MID, VULKAN_PALE, VULKAN_DARK = "#4493F8", "#79C0FF", "#1C3A5E", "#A5D6FF"
    AMBER, AMBER_PALE = "#E3A008", "#4D3B0F"
    TEAL, TEAL_PALE = "#2FBFAE", "#12443F"
    GREEN = "#3FB9A8"
    VIOLET = "#A371F7"
    PURPLE, BROWN, ORANGE = "#BC8CFF", "#D9884A", "#FF9A52"
    INK = "#E6EDF3"
    INK_SOFT = "#C9D1D9"
    GREY = "#9198A1"
    GREY_MID = "#6E7681"
    GREY_PALE = "#30363D"
    NOTE = "#9198A1"
    GRID = "#21262D"
    ROCM_BAND, VULKAN_BAND = "#24161A", "#121D2F"
    BAND_ALPHA = 0.12
    REDS = ["#5C2226", "#8E1F24", "#B62324", "#DA3633", "#F85149", "#FFA198"]
    ON_BAR = "#FFFFFF"


def out(base):
    """The directory a script should write to for the current theme."""
    d = os.path.join(base, "dark") if DARK else base
    os.makedirs(d, exist_ok=True)
    return d


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 200,
        "axes.edgecolor": GREY_MID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "normal",
        "axes.titlecolor": INK,
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "legend.frameon": False, "legend.labelcolor": INK,
        "hatch.color": GREY_MID,
    })
