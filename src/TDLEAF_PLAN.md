# TDLeaf(λ) NNUE Learning — Implementation Reference

## Overview

TDLeaf(λ) is a temporal-difference reinforcement learning algorithm adapted for minimax
search (Baxter, Tridgell, and Weaver, 2000).  It uses the game result and the sequence
of NNUE evaluations at PV leaf positions to form TD errors, then backpropagates those
errors through the NNUE network to update weights.

Build with:

```sh
perl comp.pl 2026_03_09a NNUE=1 TDLEAF=1
```

All learning code is gated by `#if TDLEAF`; when `TDLEAF=0` (default) no overhead is added.

---

## Algorithm

For a game of T half-moves:

- `d_t` = sigmoid of the **NNUE static evaluation at the PV leaf position** at ply t,
  from White's perspective:
  `d_t = 1 / (1 + exp(-score_white_t / K))`, K ≈ 400 cp.
- `z` = game result from White's perspective: 1.0 = White wins, 0.5 = draw, 0.0 = Black wins.

**Temporal difference errors (backward view):**

```
e_{T-1} = z - d_{T-1}
e_t     = (d_{t+1} - d_t) + λ * e_{t+1}     for t = T-2 … 0
```

**Weight update (gradient ascent on prediction accuracy):**

```
Δw = α * Σ_t  e_t * ∇_w d_t
```

where `∇_w d_t = d_t * (1 - d_t) / K * ∇_w score_t`.

Defaults: `λ = 0.7`, `K = 400`, `TDLEAF_ALPHA = 200` (FC+FT), `NNUE_FT_LR_SCALE = 1.0` (FT accumulator), `NNUE_PSQT_LR_SCALE = 1000` (PSQT).

**Key design choice:** `d_t` is computed from `nnue_evaluate()` (direct static eval of the
PV leaf), not from the search score propagated from the root.  This ensures the sigmoid
value and the forward-pass gradient are computed from the same NNUE evaluation of the same
position, making the gradient self-consistent.

---

## Scope: All NNUE Weights Are Trained

| Layer | Parameters | Notes |
|-------|-----------|-------|
| FC0 weights/biases | 1,024×16 int8 + 16 int32, ×8 stacks | Quantized int8, float shadow |
| FC1 weights/biases | 32×32 int8 + 32 int32, ×8 stacks | Same |
| FC2 weights/bias   | 32 int8 + 1 int32, ×8 stacks | Same |
| FT weights         | 22,528×1,024 int16 | Sparse update; float shadow on heap (~92 MB) |
| PSQT weights       | 22,528×8 int32 | Sparse update; float shadow (~720 KB) |

FT and PSQT are updated sparsely: only the ~30–60 feature rows active at each leaf
position are touched.  `ft_dirty[FT_INPUTS]` tracks which rows received gradient during
the game; only dirty rows are scanned in `nnue_apply_gradients`.

---

## File Structure

| File | Contents |
|------|----------|
| `src/tdleaf.h` | Hyperparameters, `TDRecord`, `TDGameRecord`, function declarations |
| `src/tdleaf.cpp` | PV walking, TD error computation, gradient backprop hooks |
| `src/nnue.cpp` | FP32 shadow arrays, forward pass, gradient accumulation, weight save/load |
| `src/nnue.h` | `NNUEActivations` struct, TDLeaf function declarations |

---

## Data Structures

### `TDRecord` (per-ply snapshot, stored in `TDGameRecord`)

```cpp
struct TDRecord {
    int16_t acc[2][NNUE_HALF_DIMS];          // accumulator at PV leaf (int16)
    int32_t psqt[2][NNUE_PSQT_BKTS];        // PSQT at PV leaf
    int     score_stm;                        // NNUE static eval at leaf, STM POV (cp)
    int     stack;                            // layer-stack index (piece_count-1)/4
    bool    wtm;                              // White to move at leaf
    int     ft_idx[2][NNUE_MAX_FT_PER_PERSP]; // active FT feature indices
    int8_t  n_ft[2];                          // active feature count per perspective
};
```

Memory: ≈ (2×2048 + 8×4 + 4+4+1 + 2×64×4 + 2) bytes × 400 plies ≈ 2.3 MB.

### `TDGameRecord`

```cpp
struct TDGameRecord {
    TDRecord plies[MAX_GAME_PLY];   // one per half-move (max 400)
    int      n_plies;               // plies recorded so far
};
```

One global instance in `game_rec`; `n_plies` reset to 0 at game start.

---

## PV Leaf Score

`tdleaf_record_ply` walks the PV from the root accumulator using `nnue_record_delta` /
`nnue_apply_delta`, then calls `nnue_evaluate(leaf_acc, leaf_wtm, pc)` to get the leaf
score.  `leaf_wtm = root_wtm XOR (pv_len & 1)` — the side to move at the leaf flips once
per ply walked.

The score is always stored from the leaf's side-to-move perspective (`score_stm`).
`tdleaf_update_after_game` converts to White's perspective as:
`score_white = leaf_wtm ? score_stm : -score_stm`.

To verify correctness at build time:

```sh
perl comp.pl 2026_03_09a NNUE=1 TDLEAF=1 TDLEAF_CHECK_SCORE=1
```

This prints `direct` (NNUE leaf eval) vs `propagated` (root score with per-ply sign flip)
for every ply, flagging differences > 300 cp.

---

## Gradient Flow

```
FC2 output (positional)
  → ∂/∂(positional) via cp_factor and wtm_sign
  → FC2 weights/bias
  → CReLU backward → FC1 pre-activation
  → FC1 weights/bias
  → dual-activation backward (SqrCReLU + CReLU on FC0 outputs 0–14)
  → FC0 pre-activation
  → FC0 weights/bias
  → SqrCReLU backward on accumulator pairs
  → g_acc[2][1024]  (gradient w.r.t. each accumulator unit per perspective)
  → FT weight rows for each active feature index (sparse)
  → PSQT weight rows for each active feature index (sparse)
```

**Learning rate scales** (defined in `src/tdleaf.h`):

- `NNUE_FT_LR_SCALE` multiplies `g_acc[p][d]` before accumulating into `grad_ft_w`.
  Value 1.0 means the same effective LR as FC (TDLEAF_ALPHA already in grad_scale).
  The backward chain naturally amplifies the gradient to a useful magnitude for int16 weights.

- `NNUE_PSQT_LR_SCALE` multiplies `grad_scale × 0.5` for the PSQT update.
  Needs a large value (~1000) because PSQT bypasses the FC backward chain entirely —
  `grad_scale = TDLEAF_ALPHA × e[t] × d(1−d)/K × cp_factor ≈ 2×10⁻⁴`, while PSQT
  weights are at int32 scale (~5,776 per pawn).  Without the amplification, per-game
  updates would be ~1/10,000 of a pawn.

- `NNUE_FC_BIAS_LR_SCALE` multiplies all FC bias gradients (grad_l0_b, grad_l1_b, grad_l2_b).
  Bias gradients lack the non-negative `l0_in[i]` factor that weight gradients carry.
  Because TDLeaf alternates `wtm_sign × e[t]` between +1 and −1, bias gradients from
  white-to-move and black-to-move plies nearly cancel across a game, leaving per-game
  net bias gradient < 0.5 int32 units — invisible after rounding.  A large multiplier
  (~1000) is needed to produce visible updates.

FT, PSQT, and FC biases all have independent LR scales; tune one at a time.

---

## Weight Persistence — `.tdleaf.bin` (version 3)

Saved at `{exec_path}nn-ad9b42354671.tdleaf.bin`.  Format:

```
[v2 header: version(4) + 8 FC stacks + per-layer weight/bias/count arrays]
[n_ft_rows(4 bytes)]
[per dirty row: fi(4) + ft_w[1024]×128 as int32[1024] + ft_cnt[1024] as uint32[1024]
                      + psqt_w[8]×128 as int32[8]    + psqt_cnt[8]  as uint32[8]]
```

Values are stored at 128× resolution (divide by 128 on load) to preserve sub-integer
drift across sessions.  Update counts enable weighted averaging of concurrent training runs.

Version 2 files (FC only) are still accepted on load; a notice is printed and the file
will be upgraded to version 3 on the next save.

---

## Concurrent File Access

Multiple EXchess instances (e.g. several parallel self-play games) can share a single
`.tdleaf.bin` safely via POSIX file locking and delta-based merging.

### Design

**Problem:** If two instances both read the file, apply their gradient updates to their
in-memory weights, and then write back, the second write silently overwrites the first
instance's changes.

**Solution:** Each instance tracks only its own accumulated weight *deltas* since the last
file write (not the full weight values).  On each write:

1. Acquire `LOCK_EX` on a companion `.tdleaf.bin.lock` file.
2. Re-read the current `.tdleaf.bin` from disk.
3. Merge: `merged_value = file_value + our_delta` for every entry.
4. Update the in-memory float shadows to the merged values; zero the deltas.
5. Write the merged content to `.tdleaf.bin.tmp`.
6. `rename(.tmp, .tdleaf.bin)` — atomic on POSIX filesystems.
7. Release the lock (close the lock-file fd).

Reads use `LOCK_SH` (multiple simultaneous readers allowed; blocked only during a write).

### Implementation Details

**Lock file:** `.tdleaf.bin.lock` is a separate companion file so locking survives the
atomic `rename()` of the main file.  The lock is held only during the re-read/write cycle
(a few milliseconds), not across the entire game.

**Delta arrays** (in `nnue.cpp`):

| Array | Size | Contents |
|-------|------|----------|
| `delta_l0_w[8][1024×16]` | 512 KB | FC0 weight deltas |
| `delta_l0_b[8][16]` | 512 B | FC0 bias deltas |
| `delta_l1_w[8][32×32]` | 256 KB | FC1 weight deltas |
| `delta_l1_b[8][32]` | 1 KB | FC1 bias deltas |
| `delta_l2_w[8][32]` | 1 KB | FC2 weight deltas |
| `delta_l2_b[8]` | 32 B | FC2 bias deltas |
| `ft_delta_f32` (heap) | ~92 MB | FT weight deltas (all rows) |
| `psqt_delta_f32` (heap) | ~720 KB | PSQT weight deltas |

Deltas are zeroed after each successful write (either on first write or after a
re-read-merge write).  `nnue_load_fc_weights()` also zeros all deltas to establish a
clean baseline.

**Update-count merging:** counts use `max(file_count, our_count)` — an approximation
of the total cross-instance update count sufficient for monitoring; exact accumulation
is not required for correctness.

### Usage with match.py

When running parallel self-play via cutechess-cli with multiple concurrent TDLEAF
instances, add `--wait MS` to `match.py` to insert a pause between games.  This reduces
contention on the `.tdleaf.bin.lock` file and gives each instance time to complete its
write cycle before the next game starts:

```sh
python3 match.py EXchess_vtrain_a EXchess_vtrain_ro -n 200 -c 4 --wait 500
```

---

## Initialization

**Default (fine-tuning):** When no `.tdleaf.bin` is found, the pretrained Stockfish 15.1
FC/FT/PSQT weights (from the `.nnue` file) are used automatically.  Training proceeds as
gradient updates from the SF15.1 starting point.

**Training from scratch:** Use `--init-nnue --write-nnue <file>` to create a randomly
initialised `.nnue` with no source file required:

```sh
perl comp.pl init_nnue NNUE=1 TDLEAF=1
./EXchess_vinit_nnue --init-nnue --write-nnue nn-fresh.nnue
```

This calls `nnue_alloc_arrays()` + `nnue_init_fp32_weights()` + `nnue_init_zero_weights()`
(despite the name, `nnue_init_zero_weights` samples from N(μ,σ) measured from SF15.1):

| Component | Distribution |
|-----------|-------------|
| FC0 weights | N(0.24, 8.43), truncated ±127 (rejection sampling) |
| FC1 weights | N(−1.10, 18.30), truncated ±127 (rejection sampling) |
| FC2 weights | N(1.10, 30.0), truncated ±127 — σ reduced from measured 76.38; see below |
| FC0/1/2 biases | N(μ,σ) measured from SF15.1 |
| FT weights (int16) | N(−0.71, 44.41) |
| FT biases (int16) | N(3.34, 96.48) |
| PSQT | Signed piece values: pawn ±5776, knight/bishop ±17328, rook ±28880, queen ±51984 |

**FC2 σ note:** The measured SF15.1 FC2 distribution (σ=76.38) is the *result* of training —
many weights are pushed near ±127 after learning.  Using σ=76.38 for random init clips
roughly 20% of samples to the int8 boundary, creating artificial spikes.  σ=30 avoids all
clipping (P(|X|>127) < 0.002% with rejection sampling) while providing enough weight
diversity to prevent pathological symmetry.  The trained FC2 distribution will naturally
widen as TDLeaf pushes high-importance weights toward their saturation limits.

All int8 weight sampling uses rejection sampling (not clipping): values outside ±127 are
discarded and redrawn.  FC biases are int32 with no range constraint; FT weights/biases are
int16 with σ << 32767, so no clipping issue arises for those layers.

Then build training binaries pointing at the new file:

```sh
perl comp.pl train_fresh NNUE=1 NNUE_NET=nn-fresh.nnue TDLEAF=1
perl comp.pl train_fresh_ro NNUE=1 NNUE_NET=nn-fresh.nnue TDLEAF=1 TDLEAF_READONLY=1
```

---

## Hooks in Existing Code

| Location | Change |
|----------|--------|
| `src/define.h` | `#ifndef TDLEAF / #define TDLEAF 0 / #endif` |
| `src/chess.h` — `game_rec` | `TDGameRecord td_game` inside `#if TDLEAF` |
| `src/nnue.cpp` — `nnue_load()` | Calls `nnue_init_fp32_weights()` inside `#if TDLEAF` |
| `src/main.cpp` — after `ts.search()` | `tdleaf_record_ply()` with root acc + PV |
| `src/main.cpp` — `game.over = 1` sites | `tdleaf_update_after_game()` |
| `src/main.cpp` — `new_game` / `setboard` | `td_game.n_plies = 0` |
| `src/EXchess.cc` | `#if TDLEAF #include "tdleaf.cpp" #endif` |
| `src/comp.pl` | `TDLEAF=1` flag → `-D TDLEAF=1` |

---

## Diagnostic Flags

| Flag | Effect |
|------|--------|
| `TDLEAF=1` | Enable all learning code |
| `TDLEAF_READONLY=1` | Load weights but skip gradient updates (inference only) |
| `TDLEAF_CHECK_SCORE=1` | Print direct vs propagated leaf score on every ply |

---

## Self-Play Driver

`run/tdleaf_selfplay.py` manages engine vs engine games, captures results, and runs the
trained binary continuously.  `run/compare_fc_weights.py` compares a `.tdleaf.bin` file
against the baseline `.nnue` and shows FC, FT, and PSQT weight statistics.
