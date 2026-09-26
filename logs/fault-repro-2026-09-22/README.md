# Trying to reproduce the fault, and finding out what the dropouts were, 2026-09-22

[`logs/fault-caught-2026-09-22/`](../fault-caught-2026-09-22/) records the rare fault firing during a
sweep that repeated qwen2.5-1.5B decode at depths 16384 and 24576, one `llama-bench` invocation per
point. It faulted on the fifth invocation. Two circumstantial things pointed the same way: the sweep
before it had produced two dropouts of about 28 percent at exactly those two depths and nowhere else,
and eight hours of soak at depth 0 had produced nothing at all.

So the loop was repeated on a fresh boot, with the same model, the same two depths and the same
alternation, to see whether it is a reproducer.

## It is not, in eight invocations

| invocation | depth | tg64 | faults |
|---|---|---|---|
| 1 | 16384 | 141.24 ± 0.95 | 0 |
| 2 | 24576 | 122.88 ± 0.48 | 0 |
| 3 | 16384 | 141.41 ± 0.71 | 0 |
| 4 | 24576 | 123.00 ± 0.55 | 0 |
| 5 | 16384 | 141.31 ± 0.72 | 0 |
| 6 | 24576 | **88.34 ± 0.29** | 0 |
| 7 | 16384 | **101.96 ± 0.38** | 0 |
| 8 | 24576 | 123.14 ± 0.51 | 0 |

The fifth invocation, the one that faulted last time, passed at the healthy rate. No fault line was
logged in the whole run. Eight invocations is not many against a fault that historically appears about
once in 200 rounds, so this does not refute the lead; it does mean the lead is not a reproducer, which
is what it was being tested as.

## The dropouts are the governor, and that part is settled

Invocations 6 and 7 are the same dropout the earlier sweep produced, and this run caught it in the act.
The spread inside each is tiny, 0.29 and 0.38, so all three samples of those invocations were slow
together: a sustained state for the length of an invocation, not one bad sample.

    oberon-governor[809]: [2026-09-22 05:11:00] GPU overheated, throttling
                                                 - Silencing future warnings of this type

That is 07:11:00 local, four seconds before invocation 6. The edge sensor read **93.0 C** and Tctl
93.6 C when checked during it, and the clock policy showed the 1000 MHz step selected. **The dropouts
are the governor dropping the shader clock from 1500 to 1000 MHz because the board is overheating.**

The size fits the mechanism, not just the timing. A pure clock ratio would be 0.667; the two
dropouts read 0.718 and 0.721 of their neighbours, a little above it, which is what a partly
memory-bound workload should do when the shader clock drops and the 450 MHz memory clock does not.
Invocation 8 recovers to 123.14 once the board has cooled, so it is transient throttling and not a
state the board stays in.

## What that costs the earlier reading

This is a correction to [`logs/fault-caught-2026-09-22/`](../fault-caught-2026-09-22/), which offered
those dropouts as circumstantial support for deep context being the trigger. They are not evidence for
that. They are evidence that back-to-back deep decode heats this board past the governor's limit in
about six minutes. on its own and is a different fact.

What survives of the lead is thinner: the fault happened during back-to-back deep decode, and eight
hours of soak at depth 0 produced nothing. That is one event against one clean soak, which is not a
rate.

## A caution for anyone measuring at depth

Deep-context decode run back to back reached 93 C in six minutes from a cold boot, where
[`logs/fedora44-thermal-2026-09-16/`](../fedora44-thermal-2026-09-16/) took twenty minutes of
continuous 8B prefill to reach 94 C and lost 4 percent of throughput. The loss here is 28 percent,
because that measurement averaged over rounds where this one catches whole invocations inside the
throttled state. A depth sweep with no cooling gaps will silently mix throttled and unthrottled points,
and the tight spread inside each invocation means nothing flags it.

## Files

`log` is the run ([`scripts/fault_repro_depth.sh`](../../scripts/fault_repro_depth.sh)), with the
governor's own log, the sensor readings and the clock policy captured immediately afterwards.

The thermal reading here was confirmed from the other side the same day: the identical loop with a
thirty-second pause between points produced no dropouts at all in 122 invocations and never went above
72 C ([`logs/depth-soak-2026-09-22/`](../depth-soak-2026-09-22/)).
