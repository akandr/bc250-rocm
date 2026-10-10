#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""The real GPU clock during each pass's decode, for logs/context-ceilings-2026-09-30/.

A pass's tg64 is the last 64/rate seconds of its run, after a fill that can take hours at a clock the
governor steps down to 1000 MHz for 5 s at a time when the GPU reaches its temperature limit. The decode
rate is only comparable at 1500 MHz, so this reads the SMU's gfxclk over that window, plus 3 s before it
and less the last 2 s of the run, in which the process frees its memory and the GPU is already idle: from
clock-1s.log (every second, from 2 October 23:25) where it exists, otherwise from clock-1min.log (seconds
at each clock per minute, from 30 September 19:22), where a window counts as clean only if every second
of every minute it touches was at 1500 MHz. Both logs come from scripts/clock_sampler.sh.

Usage: decode_window_clock.py logs/context-ceilings-2026-09-30 > decode-windows.txt
"""
import datetime, lzma, os, re, sys

D = sys.argv[1]
TZ = datetime.timezone(datetime.timedelta(hours=2))   # the campaign log is in CEST
RUN = re.compile(r"\[(\S+ \S+)\] (\S+) (\S+) (\S+) d(\d+): ok ([0-9.]+) (\d+)s")


def lines(name):
    p = os.path.join(D, name)
    return lzma.open(p + ".xz", "rt") if os.path.exists(p + ".xz") else open(p)


sec = {}
for l in lines("clock-1s.log"):
    f = l.split()
    if len(f) == 3:
        sec[int(f[0])] = (int(f[1]), int(f[2]) / 100)
first_sec = min(sec)
minute = {}
for l in lines("clock-1min.log"):
    f = l.split()
    minute[int(f[0])] = {int(k[:-3]): int(v) for k, v in (x.split("=") for x in f[1:] if x.endswith(tuple("0123456789")) and "MHz=" in x)}

print("pass                                       t/s   window  source  seconds at 1500 / below 1450")
for line in open(os.path.join(D, "results.log")):
    m = RUN.match(line)
    if not m:
        continue
    end = int(datetime.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp())
    rate = float(m[6]); lo = end - 64 / rate - 3; end -= 2
    name = f"{m[2]} {m[3]} {m[4]} d{m[5]}"
    if lo >= first_sec:
        win = [sec[t] for t in range(int(lo), end + 1) if t in sec]
        hi = sum(c >= 1490 for c, _ in win); low = sum(c < 1450 for c, _ in win)
        print(f"{name:42} {rate:6.2f} {len(win):5} s  1 s     {hi} / {low}")
    else:
        mins = [t for t in minute if t + 60 > lo and t <= end]
        if not mins:
            print(f"{name:42} {rate:6.2f}            no clock log yet")
            continue
        at = {}
        for t in mins:
            for c, n in minute[t].items():
                at[c] = at.get(c, 0) + n
        hi = sum(n for c, n in at.items() if c >= 1490); low = sum(n for c, n in at.items() if c < 1450)
        print(f"{name:42} {rate:6.2f} {len(mins):3} min  1 min   {hi} / {low}" + ("" if low == 0 else "  (minutes include fill)"))
