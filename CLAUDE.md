# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

Leaf is a C++ chess engine (GPL v3) by Daniel C. Homan, originally developed as EXchess (1997–2017).
The 2026 restart adds NNUE evaluation (Stockfish 15.1–compatible HalfKAv2_hm architecture) and
TDLeaf(λ) online learning from self-play.  xboard/CECP protocol; no UCI.

---

## Build System

Compilation is managed by `src/comp.pl`.  The build uses a **unity build** pattern: `src/Leaf.cc`
includes every other `.cpp` file as a single translation unit.  Binaries land in `run/`.

```sh
# Classical eval (no NNUE)
perl src/comp.pl <version>

# NNUE eval
perl src/comp.pl <version> NNUE=1

# NNUE + TDLeaf(λ) training
perl src/comp.pl <version> NNUE=1 TDLEAF=1

# Read-only weights (inference only)
perl src/comp.pl <version> NNUE=1 TDLEAF=1 TDLEAF_READONLY=1

# Skip interactive overwrite prompt
perl src/comp.pl <version> NNUE=1 OVERWRITE
```

Binary naming: `run/Leaf_v<version>` — e.g. `Leaf_v2026_03_09a`, `Leaf_vtrain_nn-fresh`.

### Key compile flags

| Flag | Effect |
|------|--------|
| `NNUE=1` | Enable NNUE evaluation |
| `TDLEAF=1` | Enable TDLeaf(λ) learning (requires NNUE=1) |
| `TDLEAF_READONLY=1` | Load `.tdleaf.bin` weights but skip updates |
| `MATERIAL_ONLY=1` | `score_pos()` returns raw material balance only |
| `NNUE_NET=<file>` | Override default network file (`nn-ad9b42354671.nnue`) |
| `OVERWRITE` | Skip overwrite prompt |
| `NATIVE=1` | Compile with `-march=native -mtune=native` (max perf, non-portable). Default uses `-march=x86-64-v3` (AVX2, portable across Intel Haswell+ and AMD Zen 1+). |

The `.nnue` network file and `.tdleaf.bin` weights file must reside in the same directory as the binary.

---

## Architecture

### Unity build (`src/Leaf.cc` include order)

`main.cpp` → `attacks.cpp` → `exmove.cpp` → `swap.cpp` → `moves.cpp` → `captures.cpp` →
`captchecks.cpp` → `hash.cpp` → `smp.cpp` → `search.cpp` → `score.cpp` →
`#if NNUE nnue.cpp` → `#if TDLEAF tdleaf.cpp` → `check.cpp` → `book.cpp` → `sort.cpp` →
`util.cpp` → `support.cpp` → `probe.cpp` → `setup.cpp` → `game_rec.cpp` →
`tree_search_functions.cpp`

Because of the unity build, LSP tools that analyse files individually will emit many false-positive
errors (unknown types, undeclared identifiers).  These are expected and can be ignored.

### Key source files

| File | Role |
|------|------|
| `src/main.cpp` | xboard protocol, game loop, TDLeaf hooks (`tdleaf_record_ply`, `tdleaf_update_after_game`, `tdleaf_replay`) |
| `src/search.cpp` | PVS alpha-beta, null-move pruning, LMR, lazy SMP, iterative deepening; tracks `id_scores[]` for TDLeaf |
| `src/score.cpp` | Classical hand-crafted eval + NNUE dispatch; NNUE/pawn/score hash probe/store |
| `src/nnue.cpp` | NNUE forward pass (int8 inference + NEON), FP32 shadow weights, gradient accumulation, `.tdleaf.bin` I/O |
| `src/tdleaf.cpp` | PV walking, TD error computation (with score-change clipping + ID-stability weighting), gradient backprop |
| `src/chess.h` | All major structs: `position`, `move`, `move_list`, `tree_search`, `game_rec` |
| `src/define.h` | Compile-time constants and flag defaults (`NNUE`, `TDLEAF`, `MATERIAL_ONLY`, piece encodings, `MAXD`, `MAX_GAME_PLY`) |
| `src/tdleaf.h` | TDLeaf hyperparameters, `TDRecord`, `TDGameRecord`, function declarations |

### NNUE architecture (HalfKAv2_hm, Stockfish 15.1–compatible)

- **Feature transformer:** 22,528 → 1,024 int16 accumulators per perspective + 8 int32 PSQT buckets
- **FC layers (×8 material-bucket stacks):** FC0 (1,024→16, SqrCReLU) → FC1 (30→32) → FC2 (32→1)
- **Score:** `(psqt_diff/2 + positional) × 100/5776` centipawns
- **Accumulators:** lazily updated at `exec_move` sites via `nnue_record_delta` / `nnue_apply_delta`
- **SIMD:** NEON (Apple M1/arm64) and AVX2/SSE4.1 (x86-64) hot paths for SqrCReLU, FC0, FC1; scalar fallback

### TDLeaf(λ) learning

- After each search, `tdleaf_record_ply()` walks the PV to the leaf, snapshots the accumulator, active feature indices, and iterative-deepening score history.
- After each game, `tdleaf_update_after_game()` computes backward TD errors (λ=0.7), applies score-change clipping (TDLEAF_SCORE_CLIP_CP=200 cp) and ID-stability weighting (TDLEAF_ID_VAR_SIGMA2=10,000 cp²), then backpropagates through FC/FT/PSQT layers.
- `tdleaf_replay()` then runs `TDLEAF_REPLAY_K` (default 1) additional passes over the last `TDLEAF_REPLAY_BUF_N` (default 8) completed games stored in a ring buffer, refreshing scores from current weights before each pass.
- Weights persist to `<net>.tdleaf.bin` (v4 format); POSIX file locking + delta merging allows concurrent multi-instance training.
- `material` in `score.cpp` is **already STM (side-to-move) POV** — do not flip it.

### Important conventions

- `material` is already from the side-to-move's perspective (see `exmove.cpp:223`).
- Piece encoding: `PAWN=1 … KING=6`; white pieces are `+8` (e.g. `WPAWN=9`).
- `NOMOVE` terminates PV arrays.
- All docs live in `docs/`; update `docs/change_log.txt` when making notable changes.

---

## Scripts

All scripts live in `scripts/`; `run/` and `learn/` have symlinks for in-place invocation.
See `docs/SCRIPT_USE.md` for full option tables.

```sh
# Run a match (from run/)
python3 scripts/match.py Leaf_vA Leaf_vB -n 200 -c 4 -tc 5+0.05

# Interactive training run (from learn/)
python3 scripts/training_run.py

# Bayesian Elo ratings (one or more PGN files combined)
python3 scripts/bayeselo_ratings.py file1.pgn file2.pgn --min 20 --report

# Remove duplicate games from PGN files
python3 scripts/pgn_dedup.py input.pgn --output deduped.pgn --report

# Visualise weight changes after training
python3 scripts/compare_nnue_learning.py learn/nn-fresh.nnue learn/nn-fresh.tdleaf.bin
```

### Training workflow summary

```sh
# 1. Initialise a fresh random network (optional)
perl src/comp.pl init_nnue NNUE=1 TDLEAF=1 OVERWRITE
./run/Leaf_vinit_nnue --init-nnue --write-nnue learn/nn-fresh.nnue

# 2. Build training binaries (symmetric self-play: both engines learn)
perl src/comp.pl train_fresh_a NNUE=1 NNUE_NET=learn/nn-fresh.nnue TDLEAF=1 OVERWRITE
perl src/comp.pl train_fresh_b NNUE=1 NNUE_NET=learn/nn-fresh.nnue TDLEAF=1 OVERWRITE

# 3. Run training matches (from learn/)
python3 match.py Leaf_vtrain_fresh_a Leaf_vtrain_fresh_b -c 5 -tc 0:03+0.05 --wait 500 -n 500
```

---

## Directory Layout

```
src/          Source code (unity-built via Leaf.cc)
docs/         Documentation (NNUE.md, TDLEAF.md, TODO.md, TRAINING.md, SCRIPT_USE.md, change_log.txt)
scripts/      Python automation scripts
run/          Compiled binaries + runtime config (search.par, opening book)
learn/        Training artifacts: .nnue, .tdleaf.bin, .games, pgn/
tools/        Third-party tools (cutechess-1.4.0/, BayesElo/)
TB/, TB_34/   Tablebase data (not in git)
testing/      Test suites and opening books
archives/     Historical EXchess source snapshots
```
