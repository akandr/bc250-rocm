# Where the faults that kill the board land, 2026-08-21

Extracted from the persistent journal with `scripts/journal_retro_sweep.sh` and
the detail lines collected in `fault_details.txt`.

## The addresses

Four of the five fatal resets left a decoded fault address. All are on `vmid 8`
through the gfxhub, client TCP (0x8):

| boot | address | offset in its 4 GiB-aligned region | R/W | perm | ring |
|---|---|---|---|---|---|
| -13 | `0x7facffff9000` | `0xffff9000`, 28672 bytes from the end | write | 0x5 | 40 |
| -13 | `0x7facffffa000` | `0xffffa000`, 24576 bytes from the end | write | 0x5 | 40 |
| -2 | `0x7f8cffff9000` | `0xffff9000`, 28672 bytes from the end | write | 0x5 | 40 |
| -2 | `0x7f8cffffa000` | `0xffffa000`, 24576 bytes from the end | write | 0x5 | 40 |
| -1 | `0x7fa800005000` | `0x5000`, 20480 bytes from the start | write | 0x5 | 40 |
| -3 | `0x7efa8ad62000` | `0x8ad62000`, mid-region | read | 0x3 | 24 |

A note on the process, added 26 August. This section used to say all four name
`llama-bench` as the faulting process. `fault_details.txt` does not carry process
names; it carries PASIDs, 2664, 18, 189 and 2460. One of those four is
independently corroborated: PASID 2460 appears in
[`../fault-hunt-2026-08-22/natural_boot0.txt`](../fault-hunt-2026-08-22/natural_boot0.txt)
against `Process llama-bench pid 1520299`. The other three cannot be resolved
from anything kept, so the process attribution is one of four rather than four of
four. It is a reasonable expectation for the rest, since llama-bench is what the
board was running, but it was not extracted with the rest of the detail.

## What stands out

Boots -13 and -2 are separate events on different days, with different PIDs and
different PASIDs, and their fault addresses share their entire low 32 bits:
`0xffff9000` and `0xffffa000` in both. The high bits, which are what address
space randomisation moves, differ. Two independent faults landing on the same
two pages of a 4 GiB-aligned region is not what a random wild pointer looks
like.

Boot -1 also sits at a 4 GiB boundary, 20 KB from the start of its region
rather than 24 to 28 KB from the end.

Boot -3 does not fit: a read rather than a write, a different permission code,
a different ring, and an offset in the middle of its region. It may be a
different failure that also ended in a reset.

## What this does not establish

Which allocation those pages belong to. The obvious check, mapping the offsets
back onto `/proc/<pid>/maps`, does not work: the faults come from processes that
no longer exist, address space layout is randomised between runs, and in a
process with multi-gigabyte mappings almost any offset falls inside one, so
containment proves nothing. Reproducing the fault while sampling the map is the
only way to name the buffer, and that needs a reproducer.

## Rate

Too low to iterate against directly. Five fatal resets across twenty retained
boots, and the one that was watched closely took 253 rounds of an eight-hour
soak to arrive. Any attempt to identify the buffer has to be built to survive
the board rebooting under it and to accumulate attempts across boots.
