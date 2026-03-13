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
e_t     = clip(d_{t+1} - d_t) + λ * e_{t+1}     for t = T-2 … 0
```

where `clip(d_{t+1} - d_t)` applies proportional scaling when the white-POV score
change between consecutive moves exceeds `TDLEAF_SCORE_CLIP_CP` centipawns — see
[Horizon Noise Mitigation](#horizon-noise-mitigation) below.

**Weight update (gradient ascent on prediction accuracy):**

```
Δw = α * Σ_t  e_t * ∇_w d_t
```

where `∇_w d_t = d_t * (1 - d_t) / K * ∇_w score_t`.

Defaults: `λ = 0.7`, `K = 400`, `TDLEAF_ALPHA = 200` (FC+FT), `NNUE_FT_LR_SCALE = 1.0` (FT weights), `NNUE_FT_BIAS_LR_SCALE = 10.0` (FT biases), `NNUE_PSQT_LR_SCALE = 1000` (PSQT), `NNUE_FC_BIAS_LR_SCALE = 1000` (FC biases).

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
| FT biases          | 1,024 int16 | Dense update; static float shadow (4 KB) |
| FT weights         | 22,528×1,024 int16 | Sparse update; float shadow on heap (~92 MB) |
| PSQT weights       | 22,528×8 int32 | Sparse update; float shadow (~720 KB) |

FT weights and PSQT are updated sparsely: only the ~30–60 feature rows active at each
leaf position are touched.  `ft_dirty[FT_INPUTS]` tracks which rows received gradient
during the game; only dirty rows are scanned in `nnue_apply_gradients`.

FT biases are updated densely every game (all 1,024 values): the gradient is the sum of
`g_acc[persp][d]` across both perspectives, amplified by `NNUE_FT_BIAS_LR_SCALE`.

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
    float   id_score_variance;               // variance of last TD_ID_HIST ID-depth scores (cp²)
    int     ft_idx[2][NNUE_MAX_FT_PER_PERSP]; // active FT feature indices
    int8_t  n_ft[2];                          // active feature count per perspective
};
```

Memory: ≈ (2×2048 + 8×4 + 4+4+1+4 + 2×64×4 + 2) bytes × 400 plies ≈ 2.3 MB.

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
  Needs a large value because PSQT bypasses the FC backward chain entirely —
  `grad_scale = TDLEAF_ALPHA × e[t] × d(1−d)/K × cp_factor ≈ 2×10⁻⁴`, while PSQT
  weights are at int32 scale (~5,776 per pawn).  Without the amplification, per-game
  updates would be ~1/10,000 of a pawn.

- `NNUE_FC_BIAS_LR_SCALE` multiplies all FC bias gradients (grad_l0_b, grad_l1_b, grad_l2_b).
  Bias gradients lack the non-negative `l0_in[i]` factor that weight gradients carry.
  Because TDLeaf alternates `wtm_sign × e[t]` between +1 and −1, bias gradients from
  white-to-move and black-to-move plies nearly cancel across a game, leaving per-game
  net bias gradient < 0.5 int32 units — invisible after rounding.  A large multiplier
  (~1000) is needed to produce visible updates.

- `NNUE_FT_BIAS_LR_SCALE` multiplies `g_acc[0][d] + g_acc[1][d]` for the FT bias gradient.
  FT biases are shared across all positions, so per-game cancellation is significant (same
  mechanism as FC biases).  The SqrCReLU backward clamp factors provide some asymmetry,
  so a moderate multiplier (~10) is likely sufficient.  Start with 10.0 and tune empirically.

FT, PSQT, FC biases, and FT biases all have independent LR scales; tune one at a time.

---

## Weight Persistence — `.tdleaf.bin` (version 4)

Saved at `{exec_path}nn-ad9b42354671.tdleaf.bin`.  Format:

```
[version(4) + 8 FC stacks: per-layer float32×128 weights/biases + uint32 counts]
[n_ft_rows(4 bytes)]
[per dirty row: fi(4) + ft_w[1024]×128 as float32[1024] + ft_cnt[1024] as uint32[1024]
                      + psqt_w[8]×128 as float32[8]    + psqt_cnt[8]  as uint32[8]]
[FT bias section (v4): ft_bias[1024]×128 as float32[1024] + ft_bias_cnt[1024] as uint32[1024]]
```

Values are stored at 128× resolution (divide by 128 on load) to preserve sub-integer
drift across sessions.  Update counts enable weighted averaging of concurrent training runs.

Version 3 files (FC + FT/PSQT, no FT bias) are still accepted on load; FT biases start
from the `.nnue` baseline and will be included on the next save.
Version 2 files (FC only) are also still accepted.  A notice is printed in both cases.

---

## Concurrent File Access

Multiple Leaf instances (e.g. several parallel self-play games) can share a single
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
| `ft_bias_delta[1024]` | 4 KB | FT bias deltas (static) |

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
python3 match.py Leaf_vtrain_a Leaf_vtrain_ro -n 200 -c 4 --wait 500
```

---

## Initialization

**Default (fine-tuning):** When no `<network>.tdleaf.bin` is found, the default network 
for that executable, `<network>.nnue`, is used as a starting point and training proceeds 
as gradient updates from that starting point.  If `<network>.tdleaf.bin` is present, corresponding 
to previous training, these values are loaded as updates to the default at startup. Any 
additional training will further update the `<network>.tdleaf.bin` file.

**Training from scratch:** Use `--init-nnue --write-nnue <file>` to create a randomly
initialised `.nnue` with no source file required:

```sh
perl comp.pl init_nnue NNUE=1 TDLEAF=1
./Leaf_vinit_nnue --init-nnue --write-nnue nn-fresh.nnue
```

This calls `nnue_alloc_arrays()` + `nnue_init_fp32_weights()` + `nnue_init_zero_weights()`:

| Component | Initialisation |
|-----------|---------------|
| FC0 weights | N(0.24, 8.43), truncated ±127 (rejection sampling) |
| FC1 weights | N(−1.10, 18.30), truncated ±127 (rejection sampling) |
| FC2 weights | N(1.10, 30.0), truncated ±127 — σ reduced from measured 76.38; see below |
| FC0/1/2 biases | **0** (zero) |
| FT weights (int16) | N(−0.71, 44.41), clipped to int16 range |
| FT biases (int16) | **0** (zero) |
| PSQT | Classical piece values (score.h): Pawn=5776, Knight=21776, Bishop=23046, Rook=34425, Queen=69144, King=0 (units: cp × 5776/100); signed ± by pside==persp |

Biases are zero-initialised because random N(μ,σ) from an unrelated SF15.1 distribution
provides no useful prior — it only adds noise that TDLeaf must overcome via its
near-cancelling per-game gradient structure.  FT weights break symmetry sufficiently for
SqrCReLU activations to be non-zero from game 1.

PSQT is initialised to classical evaluator piece values (score.h `value[]`) converted to
NNUE int32 units.  This gives TDLeaf a neutral but principled starting point rather than
random values with no positional content.

**FC2 σ note:** The measured SF15.1 FC2 distribution (σ=76.38) is the *result* of training —
many weights are pushed near ±127 after learning.  Using σ=76.38 for random init clips
roughly 20% of samples to the int8 boundary, creating artificial spikes.  σ=30 avoids all
clipping (P(|X|>127) < 0.002% with rejection sampling) while providing enough weight
diversity to prevent pathological symmetry.  The trained FC2 distribution will naturally
widen as TDLeaf pushes high-importance weights toward their saturation limits.

All int8 weight sampling uses rejection sampling (not clipping): values outside ±127 are
discarded and redrawn.

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
| `src/chess.h` — `tree_search` | `int id_scores[TD_ID_HIST]; int id_score_count;` (TD_ID_HIST=4) for ID history |
| `src/nnue.cpp` — `nnue_load()` | Calls `nnue_init_fp32_weights()` inside `#if TDLEAF` |
| `src/search.cpp` — search start | `id_score_count = 0;` reset at the start of each search |
| `src/search.cpp` — after each ID iteration | Appends current `g` to `id_scores[]` ring, increments `id_score_count` |
| `src/main.cpp` — after `ts.search()` | `tdleaf_record_ply()` with root acc + PV + `id_scores` + `id_score_count` |
| `src/main.cpp` — `game.over = 1` sites | `tdleaf_update_after_game()` then `tdleaf_replay()` |
| `src/main.cpp` — `new_game` / `setboard` | `td_game.n_plies = 0` |
| `src/Leaf.cc` | `#if TDLEAF #include "tdleaf.cpp" #endif` |
| `src/comp.pl` | `TDLEAF=1` flag → `-D TDLEAF=1` |

---

## Epoch-Based Replay

After `tdleaf_update_after_game()` applies the live gradient pass, `tdleaf_replay()`
runs `TDLEAF_REPLAY_K` (default 1) additional passes over the last `TDLEAF_REPLAY_BUF_N`
(default 8) completed games stored in a static ring buffer.

### How it works (Flavor B)

1. The completed `TDGameRecord` (accumulator snapshots + feature indices) is pushed into
   the ring buffer, replacing the oldest entry when full.
2. For each replay pass, iterate over all buffered games oldest-first:
   a. `tdleaf_refresh_scores()` rewrites each ply's `score_stm` by calling
      `nnue_evaluate_acc_raw()` on the stored `acc[][]` against the **current quantized
      weights**.  The accumulators themselves are frozen (Flavor B limitation — the leaf
      positions themselves are not re-searched).
   b. `tdleaf_accumulate_game()` computes TD errors and accumulates gradients exactly
      as in the live pass.
3. After all games in the pass are processed, `nnue_apply_gradients()` and
   `nnue_requantize_fc()` are called once, so the next pass's score refresh sees
   the updated weights.
4. Weights are saved to `.tdleaf.bin` after all K passes complete.

Score-change clipping and ID-stability weighting apply identically in replay passes
(the stored `id_score_variance` values are reused unchanged).

### Ablation results (K vs. Elo gain)

| K | Result |
|---|--------|
| 0 | Baseline — much weaker |
| 1 | **Current default — nearly as strong as K=2, more conservative** |
| 2 | Marginally better than K=1 in initial ablation; reduced to K=1 for stability |
| 3 | Slightly worse than K=2 |
| 6 | Large regression |

### Build flags

| Flag | Default | Effect |
|------|---------|--------|
| `TDLEAF_REPLAY_K` | 1 | Replay passes per game; 0 disables replay |
| `TDLEAF_REPLAY_BUF_N` | 8 | Ring buffer capacity (~4.5 MB × N static BSS) |

---

## Diagnostic Flags

| Flag | Effect |
|------|--------|
| `TDLEAF=1` | Enable all learning code |
| `TDLEAF_READONLY=1` | Load weights but skip gradient updates (inference only) |
| `TDLEAF_CHECK_SCORE=1` | Print direct vs propagated leaf score on every ply |

---

## Self-Play Driver

`scripts/training_run.py` manages the process of creating the necessary binaries
(if needed), specifies a baseline `.nnue` file, and sets up training matches.

`scripts/compare_nnue_learning.py` compares a `.tdleaf.bin`
file against the baseline `.nnue` and shows FC, FT, and PSQT weight statistics.

---

## Horizon Noise Mitigation

### Problem

TDLeaf uses consecutive leaf scores to form TD errors.  When the score changes
dramatically from one ply to the next (e.g., 300+ cp), it is often because the
*next* position falls into a tactical sequence that lies beyond the current search
horizon — a tactic the current position's evaluator cannot see.  Treating that
large score jump as a genuine evaluation signal distorts the gradient: the network
is penalised for correctly evaluating a position it cannot see past.

### Approach 1 — Score-change clipping (TDLEAF_SCORE_CLIP_CP)

When the white-POV score change between consecutive moves exceeds
`TDLEAF_SCORE_CLIP_CP` (default 200 cp), the `d[t+1] - d[t]` contribution to the
eligibility trace is scaled down *proportionally* so the effective change is capped
at 200 cp-equivalent:

```
delta_d  = d[t+1] - d[t]
delta_cp = |score_white[t+1] - score_white[t]|
if delta_cp > TDLEAF_SCORE_CLIP_CP:
    delta_d *= TDLEAF_SCORE_CLIP_CP / delta_cp
e[t] = delta_d + λ * e[t+1]
```

This preserves the *direction* of the update while reducing its magnitude when the
score swing is large.  Set `TDLEAF_SCORE_CLIP_CP` to a very large value (e.g., 1e6)
to disable this approach.

### Approach 2 — Iterative-deepening stability weighting (TDLEAF_ID_VAR_SIGMA2)

The last `TD_ID_HIST = 4` iterative-deepening scores are tracked in
`tree_search::id_scores[]`.  At each ply, their variance is stored in
`TDRecord::id_score_variance` (units: cp²).  During the update, the gradient scale
is multiplied by a soft weight:

```
id_weight = 1 / (1 + id_score_variance / TDLEAF_ID_VAR_SIGMA2)
grad_scale *= id_weight
```

Positions with stable ID scores (low variance) receive full weight; positions whose
score fluctuated across search depths are down-weighted.  `TDLEAF_ID_VAR_SIGMA2`
(default 10 000 cp²) is the reference variance — a position with variance equal to
`TDLEAF_ID_VAR_SIGMA2` receives half weight.  Set `TDLEAF_ID_VAR_SIGMA2` to a very
large value to disable this approach.

### Tuning guidance

| Hyperparameter | Default | Effect of increasing | Effect of decreasing |
|----------------|---------|---------------------|---------------------|
| `TDLEAF_SCORE_CLIP_CP` | 200 cp | Less clipping; more sensitive to large swings | More aggressive attenuation of large score changes |
| `TDLEAF_ID_VAR_SIGMA2` | 10 000 cp² | More tolerant of unstable ID scores | Stronger down-weighting of ID-unstable positions |

Both approaches are active simultaneously by default.  Use the ablation plan in
`docs/TODO.md` to isolate their individual contributions.  A good starting ablation:
run 500 games with each configuration and compare Elo gain per game vs. the baseline.
