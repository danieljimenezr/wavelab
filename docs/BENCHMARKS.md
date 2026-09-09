# Real measurements (not reasoned ones)

The plan admitted its performance figures were *reasoned, not measured*. These are measured.

## M5 — Apple M5, 10 cores, 32 GB, macOS 26.6.2, CPython 3.13.15

Date: 2026-09-08 · numpy 2.5.3 · TA-Lib 0.7.1

| n bars | TA-Lib battery (18 indicators) | ATR-ZigZag (pure Python, O(n)) |
|-------:|------------------------------:|-------------------------------:|
|   5,000 |    0.28 ms  (p95 0.32) |     1.19 ms  (p95 1.21) |
|  20,000 |    1.27 ms  (p95 1.42) |     4.88 ms  (p95 5.91) |
| 100,000 |    6.93 ms  (p95 7.26) |    24.51 ms  (p95 29.27) |
| 500,000 |   34.05 ms  (p95 34.32) |   122.30 ms  (p95 122.80) |

### Corrections to what the plan said

- The synthesis claimed "~5 ms over 500k bars in pure Python" for the ZigZag. **It is 122 ms: 25x optimistic.**
- The completeness critic estimated "200-400 ms" for the same thing. **That is 2-3x pessimistic.**
- Over the real working window (5,000 bars) the plan's 5 ms budget is met with **18x of headroom**
  on the battery and **4x** on the ZigZag.

### Design consequences

1. **numba is ruled out for good**, and by measurement, not by hunch. Recomputing everything from
   scratch on every bar is comfortably viable.
2. Offline re-parsing over 9 years of 1m data (~4.8M bars) will cost ~1.2 s of ZigZag and ~0.3 s of
   battery: perfectly acceptable as batch work.
3. **Projection to the VPS** (x86_64, ~2-3x slower single-threaded than an M5): the battery over
   5,000 bars would land around ~0.8 ms and the ZigZag around ~3.5 ms. Still not a problem. Still to
   be measured *in situ* in M0b, and the node's real table recorded here.
