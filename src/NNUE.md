# EXchess NNUE Evaluation — Implementation Notes

## Overview

EXchess can use a Stockfish-compatible NNUE (Efficiently Updatable Neural Network) for
position evaluation in place of its classical hand-crafted eval. The implementation
supports the **HalfKAv2_hm** format used by Stockfish 15/16 era networks.

Build with NNUE enabled:

```sh
g++ -o EXchess src/EXchess.cc -O3 -D VERS="dev" -D TABLEBASES=1 -D NNUE=1 -pthread
```

or via the build script:

```sh
perl comp.pl 2026_03_02b NNUE=1
```

The network file `nn-ae6a388e4a1a.nnue` must be in the same directory as the binary, or
in the directory from which the engine is launched.  This file is the default Stockfish
15.1 network and can be downloaded from:
https://github.com/official-stockfish/networks

When NNUE is not compiled in (`-D NNUE=0` or omitted), the classical eval is used
unchanged.  When NNUE is compiled in but the network file is not found, the engine falls
back to classical eval automatically.

---

## Network Architecture (HalfKAv2_hm)

| Component | Detail |
|-----------|--------|
| Feature set | HalfKAv2_hm: 32 king-buckets × 704 piece-square indices = **22,528 features** |
| Feature transformer (FT) | 22,528 → 1,024 int16 per perspective + 8 int32 PSQT per perspective |
| Layer stacks | 8 stacks selected by material bucket (piece_count / 4), each: |
| FC0 | 3,072 → 16 (dual-activation input: 2 × 1,536 = 3,072) |
| FC1 | 30 → 32 (dual-activation of FC0 outputs 0–14) |
| FC2 | 32 → 1 (output; FC0 output-15 adds directly as passthrough) |
| Activation | SqrCReLU on pairs + CReLU on full range |

**King orientation (HalfKAv2_hm):** each perspective horizontally mirrors the board
when the own king is on the queen side (files a–d), so the king always appears on files
e–h.  The EXchess convention uses `orient = ((ksq_f & 7) < 4) ? 7 : 0` where `ksq_f`
is the king square after rank-flip for the BLACK perspective.

---

## Score Calibration

| Constant | Value | Meaning |
|----------|-------|---------|
| `NNUE_CP_SCALE` | 128 | `network_out / 128` → EXchess centipawns |
| `NNUE_PSQT_SCALE` | 64 | `psqt_out / 64` → EXchess centipawns (≈ 6,527 PSQT units per pawn) |
| `NNUE_FT_SHIFT` | 6 | int16 accumulator → int8 input for dual-activation |
| `NNUE_SQR_SHIFT` | 7 | SqrCReLU product right-shift (max product = 127²=16,129 → ≤ 126) |
| `NNUE_WEIGHT_SHIFT` | 6 | FC layer outputs → [0, 127] int8 range |

`nnue_evaluate()` returns a score from the **side-to-move's perspective** (positive =
good for the side to move).  The score hash stores values in **White's perspective**;
conversion at store/retrieve: `score_w = wtm ? score : -score`.

---

## Files Added / Modified

| File | Change |
|------|--------|
| `src/nnue.h` | New: architecture constants, `NNUEAccumulator` struct, public interface |
| `src/nnue.cpp` | New: FT load/update, FC0–FC2 forward pass, NEON optimizations |
| `src/define.h` | Added `#ifndef NNUE / #define NNUE 0 / #endif` guard |
| `src/chess.h` | Added `NNUE_ACC_PARAM/DEF/ARG/NULL` macros; `NNUEAccumulator acc` in `search_node`; updated `score_pos` declaration |
| `src/score.cpp` | Added NNUE branch at top of `score_pos`: score-hash probe/store, dirty-accumulator refresh, `nnue_evaluate` call |
| `src/search.cpp` | Added accumulator init at search root (with forced dirty=true), copy+update at all three `exec_move` sites, `NNUE_ACC_ARG` at `score_pos` call sites |
| `src/main.cpp` | Added `nnue_load()` call at startup; fixed `score` command to build a temporary accumulator |
| `src/EXchess.cc` | Added `#if NNUE #include "nnue.cpp" #endif` to unity build |

---

## Optimizations Applied

### 1. Score hash integration (+22% NPS, 528K → 646K)

The NNUE branch in `score_pos` probes the existing `score_table` before calling
`nnue_evaluate`, and stores results after.  About 26–38% of evaluation calls are served
from the hash, avoiding both the forward pass and the dirty-accumulator refresh.

### 2. FC0 vdotq reordering (+7% NPS, 646K → 692K)

FC0 weights are reordered at load time to a "vdotq-friendly" layout:

```
l0_weights[s][ib*64 + ob*16 + k*4 + j]
```

where `ib = i/4`, `j = i%4`, `ob = o/4`, `k = o%4`.  The forward pass uses four
`vdotq_s32` NEON calls per 4-input block, accumulating into all 16 output registers
simultaneously (vs. 16 separate passes the compiler would generate).  Requires
`-D __ARM_FEATURE_DOTPROD` (available automatically on Apple M1/M2/M3 with the system
toolchain; no `-march=native` needed).

### 3. NEON fused dual-activation + vdotq FC1 (+4% NPS, 692K → ~720K)

**Dual activation (step 1):** The two scalar loops producing 3,072 int8 values from the
int16 accumulator were replaced with a single fused NEON loop per perspective.  One pass
reads `a[0..511]` and `a[512..1023]` together and writes all three output slices
(SqrCReLU and both CReLU halves) in 64 `int16x8` iterations instead of 1,536 scalar
iterations.

**FC1 vdotq (step 4):** FC1 weights (32×32 int8) are reordered at load time using the
same `ib*128 + ob*16 + k*4 + j` scheme.  The forward pass uses 8 `int32x4` accumulators
and 8 iterations of 8 `vdotq_s32` calls, replacing the 32×32 scalar loop.

### 4. Root-accumulator dirty fix (correctness, ~0% NPS)

At both search-root initialisation sites in `search.cpp` the accumulator dirty flags are
explicitly forced to `true` before calling `nnue_init_accumulator`.  Without this, if the
engine searched a position and then the game advanced to a new position, the stale
accumulator values from the previous position would be silently reused (dirty flags were
still `false` from the previous search).

---

## NPS Benchmarks

8-second `analyze` from the starting position, Apple M1 (arm64), single thread:

| Binary | NPS | Notes |
|--------|-----|-------|
| classical (no NNUE) | 1,645,247 | baseline |
| v2026_03_01b | 528,348 | NNUE, no optimizations |
| v2026_03_01c | 645,539 | + score hash (+22%) |
| v2026_03_01e | 691,200 | + score hash + vdotq FC0 (+31% total) |
| v2026_03_02b | ~720,000 | + NEON dual-act + vdotq FC1 + root dirty fix (+36% total) |

Remaining gap vs. classical: **~2.3×**.  At a 1-minute time control the NNUE version
typically searches approximately 2 plies shallower than the classical version.

---

## Known Issues and Remaining Work

### High priority

#### 1. Eager copy-make for the accumulator (primary remaining speed bottleneck)

The current search uses copy-make for the accumulator identical to the position:

```cpp
next->pos = pos;          // 256 bytes
next->acc = acc;          // 4,162 bytes  ← 16× more data
exec_move(smove, ply);
nnue_update_accumulator(next->acc, pos, next->pos, smove);  // ft_weights access
```

Both the 4KB copy **and** the `nnue_update_accumulator` call (which accesses the 46 MB
`ft_weights` table) happen at **every node**, even nodes that are cut off by alpha-beta or
the hash table before evaluation is ever called.  In the benchmark above, only ~42% of
nodes actually call `nnue_evaluate`; the remaining ~58% do useless accumulator work.

**Fix: lazy accumulator evaluation.**  Instead of copying and updating the accumulator at
every node, only record the feature-index changes (a few integers per move).  Materialise
the accumulator only when `score_pos` is actually called, by walking back to the nearest
already-computed ancestor and applying the buffered deltas.  This eliminates the 4KB copy
and `ft_weights` access at all cut nodes (~58% of nodes in this benchmark), potentially
yielding a further 20–35% NPS improvement.

This is the standard approach in Stockfish (SfKA lazy accumulator, introduced ~2022) and
is the most impactful remaining optimisation.

#### 2. Incremental accumulator drift (correctness investigation pending)

When a "dirty-check" optimisation was attempted (skipping the full refresh of the
non-dirty perspective in `nnue_init_accumulator`), the search produced measurably
different results (different extension counts, null-cut counts, and PV lines).  The
observed signature — more extensions, fewer null-move cutoffs, slightly different scores —
is consistent with small errors accumulating in the incrementally updated perspective over
many nodes, errors which the current "always refresh both perspectives" behaviour
silently corrects.

The root cause has not been isolated.  Suspect areas:
- `halfkav2_feature()` returning inconsistent indices under some move type/orientation combination
- Integer overflow in the int16 accumulator under unusual material configurations

Until this is resolved, `nnue_init_accumulator` should continue to refresh **both**
perspectives whenever called (current behaviour).  A future fix could enable the
dirty-check optimisation, saving roughly half the lazy-refresh work.

### Medium priority

#### 3. Search parameter tuning

The search's pruning parameters (null-move margins, futility thresholds, aspiration
windows, LMR reduction tables) were tuned for the classical eval.  The NNUE eval has a
different score distribution and may benefit from re-tuning these constants.  CLOP or a
self-play tournament with systematic variation would be the appropriate approach.

#### 4. FC2 and PSQT minor improvements

- **FC2** (32×1 output layer) is currently scalar.  With `vdotq` it would be 8
  iterations — negligible gain but trivial to add.
- **PSQT accumulator** (8 int32 values per perspective): the add/sub operations in
  `add_feat`/`sub_feat` are scalar loops of length 8, likely already unrolled by the
  compiler; no action needed.

### Low priority / informational

#### 5. Network file version

The current network `nn-ae6a388e4a1a.nnue` is the Stockfish 15.1 default.  More recent
Stockfish networks (Stockfish 16 and later) use the same HalfKAv2_hm architecture and
file format, so any official Stockfish network with the same architecture (verify by
checking the architecture string in the header) should load correctly.

#### 6. Pawn hash unused under NNUE

The classical eval stores pawn structure scores in a pawn hash table for reuse.  The NNUE
eval bypasses the classical eval entirely, so `pawn hash hits` is always 0 in NNUE mode.
The pawn hash memory (≈19 MB) is wasted when using NNUE.  Disabling or shrinking it at
build time when `NNUE=1` would recover that memory, but has no effect on playing strength.

#### 7. Multi-thread accumulator correctness

The SMP search allocates one `ts_thread_data` per thread, each with its own
`search_node n[MAXD+1]` stack including per-node accumulators.  Each thread's root
accumulator is independently initialised in the startup loop.  Thread interactions have
not been tested under NNUE; correctness is expected but unverified with THREADS > 1.
