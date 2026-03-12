# Epoch TODO

Planned investigations, improvements, and open questions.

---

## TDLeaf(λ) Training

### Learning rate tuning
The current LR scales are initial guesses and have not been systematically tuned.
PSQT in particular appears to learn very slowly — `NNUE_PSQT_LR_SCALE` (currently 1000)
may need to be raised further, or the grad computation for PSQT reviewed.  FT bias
(`NNUE_FT_BIAS_LR_SCALE = 10`) and FC bias (`NNUE_FC_BIAS_LR_SCALE = 1000`) scales
similarly benefit from empirical testing.  A short grid search varying each scale
independently across 500–1000-game runs would establish good defaults.

### Epoch-based replay for TDLeaf training

The current TDLeaf implementation is fully online: weights are updated after
each game and the game record is discarded.  A cheap alternative is to store
each game's leaf positions after self-play and make K additional passes through
them, recomputing the NNUE evaluation at each stored leaf with the updated
weights and applying the TD error on each pass.  The search is not re-run —
only the forward and backward NNUE passes at the fixed stored positions.

**Cost estimate (Flavor B — fixed leaves, recompute NNUE only):**
- Per-position: ~1–5 μs forward pass + ~10–15 μs backward pass
- Per-game (~80 leaf positions): ~1 ms per epoch
- 4,000 games × 10 epochs ≈ 40 seconds of additional compute — negligible

**Memory:**
- Storing board states only (accumulator recomputed each pass): ~160 MB for 4,000 games
- Storing accumulators (2 × 1024 int16 per position): ~1.3 GB for 4,000 games

**Caveats:**
- TDLeaf is on-policy: the stored leaf positions were found by an earlier
  version of the network.  Re-using them with updated weights is an off-policy
  approximation.  Empirically (cf. experience replay in DQN) this works well
  for small K, but degrades as the stored positions become stale relative to
  the current network.  K = 2–5 is a reasonable starting range; beyond ~5
  epochs over the same data instability is likely.
- The leaf *positions* themselves would change if the search were re-run with
  updated weights.  Flavor B accepts this as an approximation in exchange for
  the large reduction in compute cost vs. re-running the full search (Flavor A),
  which would cost the same as replaying all games × number of epochs.

**✓ Implemented (2026-03-11).** Flavor B is live with `TDLEAF_REPLAY_K` and
`TDLEAF_REPLAY_BUF_N=8` configurable at build or run time.

**Ablation results:**

| K | Result |
|---|--------|
| 0 | Baseline (no replay) — much weaker |
| 2 | **Best — adopted as default** |
| 3 | Slightly worse than K=2 |
| 6 | Large regression |

K=2 was marginally better than K=3 and substantially better than K=0.  K=6
caused a large regression, consistent with the expected instability when stored
positions become stale relative to the current network.  The default is set to
K=2 in `src/define.h`.

### Horizon noise mitigation — ablation testing plan

Two mechanisms were added to reduce the influence of tactics-beyond-horizon on TD
errors: score-change clipping (`TDLEAF_SCORE_CLIP_CP`) and ID-stability weighting
(`TDLEAF_ID_VAR_SIGMA2`).  Their contributions should be isolated before committing
to the combined defaults.

**Recommended ablation (500 games per arm, same starting network):**

| Arm | TDLEAF_SCORE_CLIP_CP | TDLEAF_ID_VAR_SIGMA2 | Description |
|-----|---------------------|---------------------|-------------|
| A (baseline) | 1e6 (disabled) | 1e6 (disabled) | Original algorithm |
| B (clip only) | 200 cp | 1e6 (disabled) | Approach 1 only |
| C (ID weight only) | 1e6 (disabled) | 10 000 cp² | Approach 2 only |
| D (combined) | 200 cp | 10 000 cp² | Both active (current default) |

**Metric:** Elo gain per game vs. the starting network (use `bayeselo_ratings.py` on
a 100-game test match against the starting network after each 500-game training run).

**Override hyperparameters at build time:**

```sh
# Arm A — no mitigation
perl comp.pl train_arm_a NNUE=1 NNUE_NET=nn-start.nnue TDLEAF=1 \
  -D TDLEAF_SCORE_CLIP_CP=1000000.0f -D TDLEAF_ID_VAR_SIGMA2=1000000.0f

# Arm B — clip only
perl comp.pl train_arm_b NNUE=1 NNUE_NET=nn-start.nnue TDLEAF=1 \
  -D TDLEAF_SCORE_CLIP_CP=200.0f -D TDLEAF_ID_VAR_SIGMA2=1000000.0f
```

(Similarly for arms C and D.)

**After the ablation:** if one approach dominates, drop the other to reduce
complexity.  If both help, the combined default is confirmed.

### ~~Bias initialisation~~ ✓ Implemented (2026-03-11)
FC biases (FC0/FC1/FC2) and FT biases are now initialised to zero in `--init-nnue` mode.
Random N(μ,σ) from the SF15.1 distribution provided no useful prior and added noise
that TDLeaf must overcome via its near-cancelling per-game gradient structure.  FT
weights still use random init so SqrCReLU activations are non-zero from game 1.

### Search parameter tuning
The search's pruning parameters (null-move margins, futility thresholds, aspiration
windows, LMR reduction tables) were tuned for the classical eval.  The NNUE eval has a
different score distribution and may benefit from re-tuning these constants.  CLOP or
a self-play tournament with systematic variation would be the appropriate approach.

---

## NNUE Infrastructure

### Pawn hash under NNUE
The classical eval stores pawn structure scores in a pawn hash table.  The NNUE eval
bypasses classical eval entirely, so `pawn hash hits` is always 0 in NNUE mode and the
pawn hash memory (≈19 MB) is wasted.  Disabling or shrinking it at build time when
`NNUE=1` would recover that memory (no effect on playing strength).

### search.par revision
Review whether `search.par` is still the right mechanism for runtime configuration —
consider what parameters are still relevant, whether the file format should be updated,
and how it interacts with xboard/cutechess invocation.

### ~~Linux compilation~~ ✓ RESOLVED (2026-03-11)
AVX2 x86-64 SIMD paths added to `nnue.cpp` for SqrCReLU, FC0, and FC1.
Fallback chain is now: NEON (ARM) → AVX2 (x86-64 with `-mavx2`) → scalar.
Default build uses `-march=x86-64-v3`; use `NATIVE=1` for CPU-native tuning.

### Multi-thread accumulator correctness
The SMP search allocates one `ts_thread_data` per thread, each with its own
`search_node n[MAXD+1]` stack including per-node accumulators.  Each thread's root
accumulator is independently initialised.  Thread interactions have not been tested
under NNUE; correctness is expected but unverified with `THREADS > 1`.
