# Is deep context the trigger? Three hours says no, and says what the dropouts were, 2026-09-22

The fault of [`logs/fault-caught-2026-09-22/`](../fault-caught-2026-09-22/) happened during a loop of
qwen2.5-1.5B decode at depths 16384 and 24576, five invocations in. Eight hours of soak at depth 0 the
same night produced nothing. That is one event against one clean run, which is a lead and not a rate,
and the reason this fault has been hard to study for a month is that nobody has had a rate for it.

[`scripts/depth_soak.sh`](../../scripts/depth_soak.sh) alternates two arms inside one run, so they
share a boot, a clock, a thermal state and an operator: the same model decoding at depth 0, and the
same model decoding at 16384 and 24576. **Both arms get the same thirty-second pause between points**,
because without it the deep arm does twice the work per round and heats the board past the governor's
limit, which would confound exactly the thing being tested.

## Result

| arm | depth | invocations | median tg64 | min | max | spread | readings below 0.95 of median |
|---|---|---|---|---|---|---|---|
| shallow | 0 | 124 | 196.61 | 191.84 | 197.86 | 3.1 % | **0** |
| deep | 16384 | 61 | 141.27 | 140.04 | 141.52 | 1.0 % | **0** |
| deep | 24576 | 61 | 123.03 | 121.87 | 123.33 | 1.2 % | **0** |

Three hours, 123 rounds of both arms, 246 invocations, **no fault line in either arm**, no new governor throttle, edge
temperature between 63 and 72 C throughout.

## Deep context is not the trigger

122 deep invocations at the two depths that faulted, against 124 shallow ones under identical
conditions, and nothing. The morning's event stays a single occurrence. Whatever turns a page fault
into a queue that will not preempt, it is not something that repeating this workload reliably produces,
and the deep-context lead should be treated as the coincidence it may well be.

This does not say the fault is rare in some new way. It has always been about one in 200 rounds of
mixed load, and 122 invocations of one narrow workload is a weaker test than that sounds. What it does
close is the specific hypothesis that these depths are a reproducer.

## The dropouts were the heat, confirmed from the other side

The same loop run back to back, with no pause, dropped 28 percent twice in seven invocations
([`logs/fault-repro-2026-09-22/`](../fault-repro-2026-09-22/)), and the governor logged
`GPU overheated, throttling` at 93 C four seconds before the first of them. Run with thirty seconds
between points, the identical loop produced **zero dropouts in 122 invocations**, and the board never
went above 72 C.

That is the thermal explanation confirmed by removing the heat, not by reading the governor's
log, which is the stronger form of it.

## Decode at depth is not noisier than decode at depth 0

The deep arms are the *tight* ones: 1.0 and 1.2 percent spread over sixty-one invocations each, against
3.1 percent for the shallow arm over 124. This bears on the open question about decode varying run to run
at depth ([INVESTIGATION.md](../../INVESTIGATION.md#open-questions)), which has been chasing that
variance through allocation, CPU pinning, hardware queues and address-space randomisation.

At a controlled temperature, over sixty-one invocations each, decode at 16384 and 24576 does not vary. The
variance that made the question is not a property of depth. It is worth saying carefully: this is one
model on one boot, the arms here are a single `-d` point each, not the ladders the earlier
measurements used, and nothing here re-measures the models where the variance was largest. But it
removes depth itself as the explanation, and the thing it leaves standing is temperature.

## Files

`log` is the run, one line per invocation with the fault count, the edge temperature and the governor's
throttle count beside each reading.
