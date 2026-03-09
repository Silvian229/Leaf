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
perl comp.pl 2026_03_07a NNUE=1
```

The network file `nn-ad9b42354671.nnue` (Stockfish 15.1 exact release, 47 MB) must be in
the same directory as the binary, or in the directory from which the engine is launched.
It can be downloaded from: https://github.com/official-stockfish/networks

When NNUE is not compiled in (`-D NNUE=0` or omitted), the classical eval is used
unchanged.  When NNUE is compiled in but the network file is not found, the engine falls
back to classical eval automatically.

---

## Network Architecture (HalfKAv2_hm)

| Component | Detail |
|-----------|--------|
| Feature set | HalfKAv2_hm: 32 king-buckets × 704 piece-square indices = **22,528 features** |
| Feature transformer (FT) | 22,528 → 1,024 int16 per perspective + 8 int32 PSQT per perspective |
| Layer stacks | 8 stacks selected by `(piece_count − 1) / 4`, each: |
| FC0 | 1,024 → 16 (SqrCReLU-only input: 512 per perspective × 2 = 1,024) |
| FC1 | 30 → 32 (dual-activation of FC0 outputs 0–14) |
| FC2 | 32 → 1 (output; FC0 output-15 adds directly as passthrough) |
| Activation | SqrCReLU (outputs 0–14) + CReLU (outputs 0–14, appended) |

**King orientation (HalfKAv2_hm):** each perspective horizontally mirrors the board
when the own king is on the queen side (files a–d), so the king always appears on files
e–h.  The EXchess convention uses `orient = ((ksq_f & 7) < 4) ? 7 : 0` where `ksq_f`
is the king square after rank-flip for the BLACK perspective.

**Own king as feature:** the own king IS included as a feature (PS_KING = 640 slot),
identical to the enemy king entry.  Both kings contribute to the accumulator for their
respective perspectives.

---

## Score Formula

The output follows the Stockfish 15.1 formula exactly:

```
value_internal = (psqt_diff / 2 + positional) / OutputScale(16)   [Stockfish Value units]
centipawns     = value_internal × 100 / NormalizeToPawnValue(361)
               = (psqt_diff / 2 + positional) × 100 / 5776
```

where:
- `psqt_diff = psqt[stm][stack] − psqt[opp][stack]`
- `positional = fc2_out + fwdOut`
- `fwdOut = fc0_raw[15] × 9600 / 8128`  (passthrough; `600×OutputScale / (127×WeightScaleBits)`)
- `5776 = 16 × 361`

Relevant quantization constants in `nnue.h`:

| Constant | Value | Meaning |
|----------|-------|---------|
| `NNUE_WEIGHT_SHIFT` | 6 | FC weight quantization: raw int32 >> 6 → [0,127] range |
| `NNUE_SQR_SHIFT` | 7 | SqrCReLU shift: `fc0_raw² >> (2×6+7=19)` → [0,127] |

**SqrCReLU detail:** the raw int32 fc0 value is squared *before* clamping:
`output = clamp(0, 127, fc0_raw² >> 19)`.  This matches Stockfish's `SqrClippedReLU`
exactly — negative fc0 values produce non-zero squared outputs.

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

### 5. Lazy accumulator evaluation (+17% NPS, ~720K → ~840K)

`nnue_apply_delta` replaces the temporary full-refresh stub with true one-level-lazy
incremental evaluation.  The search wiring was already in place (v2026_03_02b added
`nnue_record_delta` calls at all `exec_move` sites and the `nnue_apply_delta` guard before
`score_pos`); this optimization completes the implementation.

**At every `exec_move`:** `nnue_record_delta` stores ≤4 feature-index integers per
perspective into `acc.add/sub` arrays (no `ft_weights` access) and sets `acc.computed=false`.

**At `score_pos` time only:** `nnue_apply_delta` copies the parent's accumulator (2KB per
perspective) and applies the stored deltas via `add_feat`/`sub_feat`.  For king moves
(`need_refresh[p]=true`), the king's own perspective is rebuilt from scratch; the opponent
perspective is still incremental.

Cut nodes — roughly 58% of all nodes — now pay only a handful of integer assignments
instead of a 4KB copy plus 1K `add_feat`/`sub_feat` calls into the 46 MB `ft_weights`
table.

### 6. King-capture lazy accumulator fix (correctness, v2026_03_06z)

When a king *captures* a piece (`Kxf7` etc.), `nnue_record_delta` was only recording the
king's movement (sub from-square, add to-square) but omitting the subtraction of the
captured piece from the opponent's incremental accumulator.  Fix: added `if (capt_pt)`
capture subtraction in the king-move branch of both `nnue_record_delta` and
`nnue_update_accumulator`.  Verified via `NNUE_CHECK_LAZY` across multiple positions
including king-capture sequences.

### 7. Own-king feature inclusion (correctness, v2026_03_07a)

`halfkav2_feature()` was returning −1 (skip) for the own king, but Stockfish includes the
own king as a feature in the PS_KING = 640 slot, the same as the enemy king.  Removing
the exclusion gives correct feature indices and accumulator values for all positions.
This was a major source of evaluation error.

### 8. SqrCReLU: square raw value before clamping (correctness, v2026_03_07a)

The dual-activation loop was computing `v = clamp(0,127, fc0>>6); sq = v²>>7` — clamping
to `[0,127]` *before* squaring, which zeroed all negative fc0 values.  Stockfish's
`SqrClippedReLU` squares the raw int32 value first:
`output = clamp(0, 127, fc0_raw² >> 19)`.  Many fc0 outputs are strongly negative (e.g.
−13,000 to −8,000 in typical middlegame positions); these now contribute 127 to FC1 input
instead of 0, dramatically improving the positional component of the evaluation.

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
| v2026_03_02c | ~840,000 | + lazy accumulator (+59% vs baseline) |
| v2026_03_02f | ~870,000 | + singular-extension accumulator fix (correctness) |
| v2026_03_07a | ~870,000 | + own-king fix + SqrCReLU fix + unified scale (correctness) |

Remaining gap vs. classical: **~1.9×**.  At a 1-minute time control the NNUE version
typically searches approximately 1–2 plies shallower than the classical version.

---

## Match Results

Self-play matches at 1 min + 0.1 s/move, 100 games each:

| Match-up | Score | Notes |
|----------|-------|-------|
| v2026_03_02g vs classical | 10.0% (0W 2D 18L) | old code, broken scales |
| v2026_03_06z vs classical | 22.5% (4W 1D 15L) | fixed lazy acc, split scale |
| v2026_03_07a vs v2026_03_06z | **98.0% (96W 4D 0L)** | own-king + SqrCReLU + unified scale |
| **v2026_03_07a vs classical** | **96.0% (92W 8D 0L)** | current best |

---

## TDLeaf(λ) Online Training

All NNUE layers (FC0/FC1/FC2, FT biases, FT weights, PSQT) can be trained via
TDLeaf(λ) self-play.  See `src/TDLEAF_PLAN.md` for the full reference.
Build with `NNUE=1 TDLEAF=1`.

The `learn/training_run.py` interactive script manages a full training run: net
selection or random init, building both a training and a read-only binary, running
N iterations of M-game self-play matches via `run/match.py`, tracking cumulative game
counts across sessions, and exporting the trained weights to a new `.nnue` file named
`<net_base>-<total_games>g.nnue`.  Fischer Random (Chess960) is the default opening
randomisation method.
Key hyperparameters (in `src/tdleaf.h`):

| Constant | Value | Notes |
|----------|-------|-------|
| `TDLEAF_ALPHA` | 200 | Learning rate for FC and FT layers |
| `NNUE_FT_LR_SCALE` | 1.0 | FT accumulator LR multiplier (no extra scale needed) |
| `NNUE_PSQT_LR_SCALE` | 1000 | PSQT LR multiplier (large: PSQT bypasses FC chain) |
| `NNUE_FC_BIAS_LR_SCALE` | 1000 | FC bias LR multiplier (wtm_sign cancellation fix) |
| `NNUE_FT_BIAS_LR_SCALE` | 10 | FT bias LR multiplier (shared-bias cancellation fix) |
| `TDLEAF_LAMBDA` | 0.7 | Eligibility trace decay |
| `TDLEAF_K` | 400 | Sigmoid temperature (cp) |

---

## Remaining Work

### Search parameter tuning

The search's pruning parameters (null-move margins, futility thresholds, aspiration
windows, LMR reduction tables) were tuned for the classical eval.  The NNUE eval has a
different score distribution and may benefit from re-tuning these constants.  CLOP or a
self-play tournament with systematic variation would be the appropriate approach.

### Pawn hash unused under NNUE

The classical eval stores pawn structure scores in a pawn hash table for reuse.  The NNUE
eval bypasses the classical eval entirely, so `pawn hash hits` is always 0 in NNUE mode.
The pawn hash memory (≈19 MB) is wasted when using NNUE.  Disabling or shrinking it at
build time when `NNUE=1` would recover that memory, but has no effect on playing strength.

### Multi-thread accumulator correctness

The SMP search allocates one `ts_thread_data` per thread, each with its own
`search_node n[MAXD+1]` stack including per-node accumulators.  Each thread's root
accumulator is independently initialised in the startup loop.  Thread interactions have
not been tested under NNUE; correctness is expected but unverified with THREADS > 1.
