# EXchess

**EXchess** is an open-source chess engine written in C++ by Daniel C. Homan, an astrophysicist at Denison University in Granville, Ohio.  Development began in the late 1990s and the engine was actively maintained and released through early 2017.  After a long hiatus, the project has been restarted in 2026 with significant new features developed in collaboration with [Claude Code](https://claude.ai/claude-code) (Anthropic).

---

## History

EXchess first appeared around 1997–1998 and was one of a handful of serious open-source engines of that era.  Over two decades of on-and-off development produced a series of increasingly capable versions — from the early v2/v3 series through the v6 and v7 lines released in 2011–2017.

The engine is written in C++, licensed under the GNU Public License, and communicates via the Chess Engine Communication Protocol (xboard/Winboard).  It includes a classical hand-crafted evaluation function, principal variation search (PVS), null-move pruning, late move reductions, static exchange evaluation, history heuristics, and a lazy SMP work-sharing implementation that achieves roughly 1.65× speedup on 2 threads and 2.5× on 4 threads.  An early form of Temporal Difference (TD-leaf) learning for evaluation tuning was present in older releases.

The final pre-hiatus release was **v7.97b** (February 2017), rated around **2,772 Elo** on CCRL 40/40.  After that, development went quiet for approximately eight years.

More history and technical background can be found on the [Chessprogramming wiki](https://www.chessprogramming.org/EXchess) and the [Daniel Homan](https://www.chessprogramming.org/Daniel_Homan) author page.

---

## 2026 Restart — Collaboration with Claude Code

In early 2026 the project was restarted with a focus on two major new capabilities: a **Stockfish-compatible NNUE evaluation** and a **TDLeaf(λ) online learning** system that can train the NNUE weights from self-play.

This work was developed interactively with [Claude Code](https://claude.ai/claude-code), Anthropic's AI coding assistant.  The collaboration covered design, implementation, debugging, and tuning — from the initial NNUE forward-pass implementation through to verifying the evaluation matched Stockfish 15.1 exactly and training the network from self-play games.

---

## New Features

### NNUE Evaluation

EXchess supports **HalfKAv2_hm** NNUE evaluation compatible with Stockfish 15.1 era networks.  Build with `NNUE=1`.

The included network file, **`nn-ad9b42354671.nnue`** (the exact Stockfish 15.1 release network, 47 MB), serves three distinct purposes in the project:

**1. Implementation correctness anchor.**
Because this is the exact network shipped with Stockfish 15.1, EXchess's forward pass can be validated against the Stockfish 15.1 source line by line.  Any discrepancy in evaluation of a given position is a bug in EXchess, not an approximation.  This property was used extensively during development: several significant bugs were isolated and fixed by comparing EXchess evaluation against Stockfish on the same position, including an incorrect feature index for the own king, a wrong SqrCReLU formulation that zeroed all negative pre-activations, and an incorrect PSQT scale factor.  After all fixes, EXchess matches Stockfish 15.1 evaluation exactly (within 1 cp rounding) on every tested position.

**2. Playing-strength baseline.**
A network trained from scratch by EXchess itself (via TDLeaf(λ) self-play) will initially be weaker than `nn-ad9b42354671.nnue`, which represents years of Stockfish training data.  Match results against the Stockfish net provide the clearest measure of training progress: the goal is to close the gap, then surpass it with a network tuned to EXchess's own search characteristics.  Current result with the SF15.1 net: **92W–8D–0L (96.0%)** vs the classical EXchess eval at 10+0.1s/move.

**3. Weight statistics for random initialisation.**
The SF15.1 net's weight distributions (means and standard deviations per layer, measured empirically) are used to initialise a fresh network when training from scratch.  EXchess can generate a randomly initialised `.nnue` with `--init-nnue --write-nnue <file>`, sampling FC weights from measured Gaussian distributions and PSQT values from signed piece-value priors.  This avoids starting from zero (which is a very poor prior for chess) while remaining independent of the Stockfish training data for weights.

| Layer | Distribution used for random init |
|-------|----------------------------------|
| FC0 weights | N(0.24, 8.43), clipped ±127 |
| FC1 weights | N(−1.10, 18.30), clipped ±127 |
| FC2 weights | N(1.10, 76.38), clipped ±127 |
| FT weights (int16) | N(−0.71, 44.41) |
| PSQT | Signed piece values: pawn ±5,776; knight/bishop ±17,328; rook ±28,880; queen ±51,984 |

The network file itself is not modified by EXchess.  All trained weights are stored in a companion **`.tdleaf.bin`** file and loaded on top of (or instead of) the base network at startup.

**Architecture summary:**

| Component | Detail |
|-----------|--------|
| Feature set | HalfKAv2_hm: 32 king-buckets × 704 piece-square indices = 22,528 features |
| Feature transformer | 22,528 → 1,024 int16/perspective + 8 int32 PSQT/perspective |
| Layer stacks | 8 stacks selected by `(piece_count − 1) / 4` |
| FC0 | 1,024 → 16 (SqrCReLU input: 512/perspective × 2) |
| FC1 | 30 → 32 (dual-activation of FC0 outputs 0–14) |
| FC2 | 32 → 1 (FC0 output-15 adds as passthrough) |
| Score formula | `(psqt_diff/2 + positional) × 100 / 5776` (Stockfish 15.1 exact) |

See [`src/NNUE.md`](src/NNUE.md) for full architecture notes, NEON optimizations, and benchmark results.

### TDLeaf(λ) Online Learning

EXchess includes a complete **TDLeaf(λ)** reinforcement learning system (Baxter, Tridgell & Weaver, 2000) that trains all NNUE layers from self-play games.  The long-term goal is for EXchess to develop its own network, tuned to its own search, entirely through self-play — experiments are already in progress.

- Trains **all layers**: FC0, FC1, FC2, the 46 MB feature transformer, and PSQT weights
- Uses PV leaf scores as the TD signal; gradients flow backward through the full NNUE forward pass
- FT and PSQT are updated **sparsely** — only the ~30–60 active feature rows per position are touched
- Weights are persisted to a companion `.tdleaf.bin` file after each game, supporting fine-tuning from the Stockfish 15.1 starting point or training from a randomly initialised network
- **Concurrent multi-instance support:** multiple engine processes can share a single `.tdleaf.bin` via POSIX file locking and per-session delta accumulation with atomic rename

Build with `NNUE=1 TDLEAF=1`.  See [`src/TDLEAF_PLAN.md`](src/TDLEAF_PLAN.md) for the full algorithm, gradient flow, file format, and hyperparameter reference.

---

## Building

EXchess uses a unity build — `src/EXchess.cc` includes all other `.cpp` files.

**Classical eval (no NNUE):**
```sh
g++ -o EXchess src/EXchess.cc -O3 -D VERS="dev" -D TABLEBASES=1 -pthread
```

**With NNUE evaluation:**
```sh
perl src/comp.pl <version> NNUE=1
# e.g.  perl src/comp.pl 2026_03_08a NNUE=1
```

**With NNUE + TDLeaf(λ) learning:**
```sh
perl src/comp.pl <version> NNUE=1 TDLEAF=1
```

The `perl comp.pl` build script handles include paths, optimization flags, and optional `OVERWRITE` to skip the interactive prompt.  Built binaries land in `run/`.

The network file `nn-ad9b42354671.nnue` must be present in the same directory as the binary (or the directory from which the engine is launched).  It can be obtained from the [official Stockfish networks repository](https://github.com/official-stockfish/networks).

---

## Running

EXchess speaks the **xboard/CECP** protocol exclusively.  Point any xboard-compatible GUI at the binary, or run it directly from the command line:

```sh
cd run/
./EXchess_v2026_03_08a
```

Self-play matches between two EXchess versions (requires [cutechess-cli](https://github.com/cutechess/cutechess)):

```sh
cd run/
python3 match.py EXchess_vA EXchess_vB -n 200 -c 4 -tc 10+0.1
```

---

## License

GNU General Public License.  See [`src/license.txt`](src/license.txt) for the full license text.

---

## Acknowledgements

- Classical search and evaluation by **Daniel C. Homan** (1997–2017, 2026–present)
- NNUE architecture and network weights from the [Stockfish](https://stockfishchess.org) project (GPL v3)
- NNUE implementation, TDLeaf(λ) learning system, and 2026 restart developed in collaboration with **[Claude Code](https://claude.ai/claude-code)** (Anthropic)
- [Chessprogramming wiki](https://www.chessprogramming.org) for algorithm references
