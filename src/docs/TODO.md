# EXchess TODO

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

### Bias initialisation
FC biases and FT biases are currently initialised from the SF15.1 distribution (random
N(μ,σ)).  Consider initialising all biases to 0 and letting TDLeaf learn them from
scratch — this removes any dependence on the SF15.1 starting point and may converge to
better values, especially for a freshly-initialised network.

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

### Multi-thread accumulator correctness
The SMP search allocates one `ts_thread_data` per thread, each with its own
`search_node n[MAXD+1]` stack including per-node accumulators.  Each thread's root
accumulator is independently initialised.  Thread interactions have not been tested
under NNUE; correctness is expected but unverified with `THREADS > 1`.
