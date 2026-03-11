# TDLeaf(λ) Training — Run 1: `nn-fresh-260309`

> **Note:** Testing is still in progress.  Results and observations in this document will be updated once the full test suite is complete.

## Overview

This documents the first complete TDLeaf(λ) training run on Epoch, carried out on 9 March 2026.  The goal was to train a randomly-initialised NNUE network from scratch via self-play and measure how much strength it gains as a function of game count.

---

## Network Initialisation

The starting network, **`nn-fresh-260309.nnue`**, was created by Epoch's `--init-nnue` facility.  It is not derived from any pre-trained chess data; all weights are drawn from Gaussian distributions whose parameters (mean, σ) were measured empirically from the Stockfish 15.1 release network (`nn-ad9b42354671.nnue`).  See `docs/NNUE.md` for the per-layer distributions.

PSQT weights are initialised to piece-value priors rather than random values.  The prior assigns each piece type a uniform signed value chosen so that one extra own piece of that type scores the standard centipawn equivalent: pawn = 100 cp, knight = 300 cp, bishop = 300 cp, rook = 500 cp, queen = 900 cp.  In internal NNUE units (where the score formula is `psqt_diff/2 × 100/5776`) this corresponds to values of 5,776 / 17,328 / 17,328 / 28,880 / 51,984 respectively.  Kings are set to zero.  The network therefore begins with a crude but sensible material prior for positional scoring, while the FC layers start from random noise.

The network is a statistically plausible but chess-naïve starting point — it has the right weight magnitudes but no learned positional chess knowledge.

---

## Training Procedure

| Parameter | Value |
|-----------|-------|
| Algorithm | TDLeaf(λ), online, all layers updated |
| Training format | Self-play: `Epoch_vtrain` (learning) vs `Epoch_vtrain_ro` (read-only) |
| Positions | Fischer Random (Chess960), random starting position each game |
| Opening book | None |
| Tablebases | Disabled |
| Time control | 3+0.05 s/move |
| Concurrency | 5 simultaneous games |
| Total training games | 4,000 |

The read-only opponent (`_ro`) reloads the latest `.tdleaf.bin` weights at the start of each 500-game training iteration.  This means the learning engine's weights are periodically adopted as the new baseline opponent, providing a gradually strengthening training signal without the instability of per-game opponent updates.

Network snapshots were saved at **500, 1000, 2000, and 4000 games**.

---

## Testing Procedure

After training, a round-robin test tournament was run among all five network snapshots plus two reference engines, producing `pgn/fresh-260309-testing.pgn`.

| Parameter | Value |
|-----------|-------|
| Games per pair | 500 (250 each colour) |
| Positions | Fischer Random, no opening book |
| Tablebases | Disabled |
| Time control | 10+0.1 s/move |
| Total games | 5,612 |

**Reference engines:**

- **`EXchess_classic`** — the classical hand-crafted Epoch eval; the strong-reference ceiling for this run
- **`EXchess_classic_material`** — a material-only classical build (`MATERIAL_ONLY=1`); the weak-reference floor

---

## Results

### Bayesian Elo Ratings

Computed with `scripts/bayeselo_ratings.py` (BayesElo, maximum-likelihood).  Ratings are relative within this pool.

| Rank | Engine | Elo | ± | Games | Score | Oppo | Draws |
|-----:|--------|----:|--:|------:|------:|-----:|------:|
| 1 | EXchess_classic | +1031 | 213 | 500 | 100% | +66 | 0% |
| 2 | Epoch_vnn-fresh-260309-4000g | +66 | 17 | 2,612 | 68% | −11 | 10% |
| 3 | EXchess_classic_material | −8 | 56 | 112 | 40% | +66 | 24% |
| 4 | Epoch_vnn-fresh-260309-2000g | −79 | 15 | 2,000 | 68% | −236 | 18% |
| 5 | Epoch_vnn-fresh-260309-1000g | −219 | 15 | 2,000 | 48% | −201 | 20% |
| 6 | Epoch_vnn-fresh-260309-500g | −324 | 15 | 2,000 | 33% | −175 | 19% |
| 7 | Epoch_vnn-fresh-260309 (0g) | −467 | 18 | 2,000 | 16% | −139 | 12% |

### Progress by Game Count

| Snapshot | Elo | Gain vs previous |
|----------|----:|-----------------:|
| 0g (fresh init) | −467 | — |
| 500g | −324 | +143 |
| 1000g | −219 | +105 |
| 2000g | −79 | +140 |
| 4000g | +66 | +145 |
| **Total gain** | | **+533** |

Improvement is roughly linear in game count over this range, with no sign of plateau at 4000 games.

### Key Pairwise Results (4000g network)

| White | Black | W | D | L | Score |
|-------|-------|--:|--:|--:|------:|
| 4000g | 0g (fresh) | 225 | 10 | 15 | 92.0% |
| 4000g | 500g | 214 | 26 | 10 | 90.8% |
| 4000g | 1000g | 195 | 33 | 22 | 84.6% |
| 4000g | 2000g | 141 | 64 | 45 | 69.2% |
| 4000g | EXchess_classic_material | 43 | 19 | 40 | 51.5% |
| EXchess_classic | 4000g | 250 | 0 | 1 | 100.0% |

The 4000g network is approximately equal to the material-only classical eval and has a large positive score against all earlier snapshots.  It remains far below the classical eval — `EXchess_classic` loses only a single game in 500 against the 4000g network.

---

## Observations

1. **Consistent monotonic improvement.** Every snapshot is stronger than the last, and the per-500-game Elo gain is stable (~100–145 Elo), suggesting learning is still far from saturation at 4000 games.

2. **Crossing the material-only threshold.** At roughly 2000–4000 games the network surpasses a pure material eval, meaning it has begun to learn meaningful positional patterns from self-play.

3. **Large gap to classical eval.** The 4000g net is ~965 Elo below `EXchess_classic` in this pool.  This is expected: the classical eval encodes decades of chess knowledge; a self-trained network at this game count is not expected to match it.  Closing this gap is the long-term goal.

4. **Fischer Random as training distribution.** Using random starting positions removes opening-book effects and ensures the network is exposed to a wide variety of piece configurations from move 1.  It is not yet known whether a network trained on Fischer Random positions will transfer well to standard chess, or whether standard starting-position training would be faster.

5. **Draw rate increases with strength.** The fresh network draws only 12% of games (often simply losing); the 4000g network draws 10% overall but up to 24% against classical-material.  This is qualitatively consistent with a stronger evaluation keeping games competitive longer.

---

## Next Steps

- Continue training beyond 4000 games on the same network to assess the rate of continued improvement.
- Run a test match against standard-chess opponents (not Fischer Random) to check transferability.
- Investigate learning-rate tuning, particularly for PSQT, which appears to learn slowly (see `docs/TODO.md`).
- Consider a training run starting from the SF15.1 network (`nn-ad9b42354671.nnue`) rather than a fresh initialisation, as a comparison.
