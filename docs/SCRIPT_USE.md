# Leaf Script Reference

All Python scripts live in `scripts/`.  Symlinks in `run/` and `learn/` allow
them to be invoked in-place from those directories, which is the normal workflow
since engines, `.nnue` files, and `.tdleaf.bin` files live there.

---

## match.py

Run a head-to-head match or gauntlet between Leaf executables using
cutechess-cli.  **Invoke from `run/`** (engines and `.nnue` files must be in the
working directory).

```sh
cd run/

# Head-to-head, 200 games
python3 match.py Leaf_vA Leaf_vB -n 200 -c 4 -tc 5+0.05

# Multi-iteration symmetric self-play (both engines learn; restart between
# iterations so each picks up the merged .tdleaf.bin weights)
python3 match.py Leaf_vtrain_a Leaf_vtrain_b -n 500 -i 10 --wait 500

# Gauntlet: probe engine vs multiple opponents; all games appended to one PGN
python3 match.py Leaf_vnew Leaf_v1 Leaf_v2 Leaf_v3 \
    -n 100 --pgn results/gauntlet.pgn
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--games` | 100 | Games per iteration per opponent |
| `-i`, `--iterations` | 1 | Iterations per opponent; engines restart between each |
| `-c`, `--concurrency` | cpu_count/2 | Simultaneous games |
| `-tc`, `--time-control` | `10+0.1` | Time control (`moves/time+inc` or `time+inc`, seconds) |
| `--pgn FILE` | — | Persistent PGN; all games appended across opponents/iterations |
| `--pgn-out FILE` | auto | Per-iteration PGN (default: `match_<e1>_vs_<e2>.pgn`) |
| `--fischer-random` | off | Chess960 starting positions |
| `--wait MS` | 0 | Milliseconds between games (useful when sharing a `.tdleaf.bin`) |
| `--depth1 N` | — | Limit engine1 search to depth N |
| `--depth2 N` | — | Limit engine2 search to depth N |
| `--openings FILE` | — | EPD or PGN openings file, randomly ordered |

When more than one opponent is supplied the script enters **gauntlet mode** and
prints a summary table (Opponent, Games, W, D, L, Score%, Elo diff) at the end.

---

## training_run.py

Interactive TDLeaf(λ) training run manager.  **Invoke from `learn/`** so that
all working files (`.nnue`, `.tdleaf.bin`, `.games`, built binaries, PGN output)
land in `learn/`.

```sh
cd learn/
python3 training_run.py
```

The script prompts for:

1. **Starting network** — existing `.nnue` file or a freshly random-initialised one
2. **Build** — compiles two training binaries (`_a` and `_b`, both `NNUE=1 TDLEAF=1`)
   via `src/comp.pl`; both write to the shared `.tdleaf.bin` (symmetric self-play)
3. **Continuity** — whether to continue from an existing `.tdleaf.bin` or start fresh
4. **Match parameters** — games/iteration, iteration count, TC, concurrency, wait, Fischer Random

Both binaries are full learners: every game produces gradient updates from both sides
of the board, doubling the signal per game.  Concurrent writes are safe via the
`flock`+delta-merge mechanism in `nnue_save_fc_weights`.  On completion the trained
weights are exported to `<net_base>-<total_games>g.nnue`.

Game counts accumulate in a `<net_base>.games` sidecar file across multiple runs on
the same network.

---

## compare_nnue_learning.py

Visualise NNUE weight changes after TDLeaf training.  **Invoke from `learn/`**
(where the `.nnue` baseline and `.tdleaf.bin` live).

```sh
cd learn/
python3 compare_nnue_learning.py nn-fresh-260309.nnue nn-fresh-260309.tdleaf.bin
```

Produces a four-page matplotlib figure:

- **Page 1 — FC weights**: FC0/FC1/FC2 weight distributions (baseline vs learned),
  per-output delta histograms, per-stack % changed + max |Δ|
- **Page 2 — FC biases**: FC0/FC1/FC2 bias distributions (baseline vs learned, int32),
  delta histograms, per-stack scatter of individual Δ values (every bias visible so
  no outlier can hide in an aggregate)
- **Page 3 — Feature transformer**: FT bias distributions (baseline vs learned, v4
  `.tdleaf.bin` only), FT weight distributions, delta and update counts
- **Page 4 — PSQT**: baseline vs learned distributions, delta histogram,
  per-bucket mean delta bar chart ±1σ

Optional flags:

```sh
# Save pages to PNG files instead of displaying
python3 compare_nnue_learning.py baseline.nnue weights.tdleaf.bin --save out_prefix

# Include full FT weight arrays (slow; requires ~92 MB of memory per perspective)
python3 compare_nnue_learning.py baseline.nnue weights.tdleaf.bin --ft-weights
```

---

## verify_fc0.py

Debugging tool: recomputes FC0 outputs from a manually specified `l0_in` vector
and compares against raw weights read directly from the `.nnue` file.  Used to
verify the Leaf forward pass against the reference network.

Edit the `L0_IN` dict near the top of the script with values captured from
`NNUE_DEBUG_VERBOSE=1` output, then run from `run/`:

```sh
cd run/
python3 verify_fc0.py
```

---

## launch_epoch.py

Registers an Leaf binary in cutechess's `engines.json` and launches the
cutechess GUI.  **Invoke from `run/`**.

```sh
cd run/
python3 launch_epoch.py Leaf_v2026_03_09a
```

If no executable name is given, defaults to `Leaf`.

---

## run_epoch.py

Wrapper that launches an Leaf binary with `run/` as its working directory,
ensuring it finds `search.par`, `main_bk.dat`, and the `.nnue` network file.
Useful as the engine command inside cutechess or other GUIs.

```sh
# As engine command in cutechess:
python3 /path/to/run/run_epoch.py Leaf_v2026_03_09a
```

---

## bayeselo_ratings.py

Compute a Bayesian Elo rating list for all players in a PGN file, using the
`tools/BayesElo/bayeselo` binary.  Can be invoked from anywhere.

```sh
# Basic usage
python3 scripts/bayeselo_ratings.py learn/pgn/fresh-260309-testing.pgn

# Exclude players with fewer than 50 games
python3 scripts/bayeselo_ratings.py results.pgn --min 50

# Also optimise first-move advantage and draw-Elo
python3 scripts/bayeselo_ratings.py results.pgn --advantage --drawelo

# Use a non-default bayeselo binary
python3 scripts/bayeselo_ratings.py results.pgn --bayeselo /usr/local/bin/bayeselo
```

Example output:

```
Bayesian Elo ratings — fresh-260309-testing.pgn
5612 games loaded, 7 players rated

Rank  Name                            Elo     ±  Games   Score   Oppo  Draws
----------------------------------------------------------------------------
   1  EXchess_classic             +1031   213    500    100%    +66     0%
   2  Leaf_vnn-fresh-260309-4000g    +66    17   2612     68%    -11    10%
   3  EXchess_classic_material         -8    56    112     40%    +66    24%
   4  Leaf_vnn-fresh-260309-2000g    -79    15   2000     68%   -236    18%
   5  Leaf_vnn-fresh-260309-1000g   -219    15   2000     48%   -201    20%
   6  Leaf_vnn-fresh-260309-500g    -324    15   2000     33%   -175    19%
   7  Leaf_vnn-fresh-260309         -467    18   2000     16%   -139    12%
```

Ratings are relative (zero-sum); absolute scale depends on which players are
included.  The `±` column is the larger of the two asymmetric confidence
intervals returned by BayesElo.  `Oppo` is the average Elo of opponents faced.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `pgn` | *(required)* | PGN file(s) to analyse (combined before analysis) |
| `--bayeselo PATH` | `tools/BayesElo/bayeselo` | Path to the bayeselo binary |
| `--min N` | 0 | Exclude players with fewer than N games |
| `--advantage` | off | Optimise first-move advantage alongside ratings |
| `--drawelo` | off | Optimise draw-Elo alongside ratings |

Multiple PGN files can be supplied; BayesElo reads them sequentially and combines all games:

```sh
python3 scripts/bayeselo_ratings.py learn/pgn/run1.pgn learn/pgn/run2.pgn --min 20
```

---

## pgn_dedup.py

Remove duplicate games from one or more PGN files.  Two games are considered
identical when their move sequences match after stripping move numbers, comments,
NAG annotations, and result tokens.

```sh
# Deduplicate a single file
python3 scripts/pgn_dedup.py input.pgn --output deduped.pgn --report

# Combine and deduplicate multiple files
python3 scripts/pgn_dedup.py run1.pgn run2.pgn run3.pgn --output combined.pgn --report
```

Example `--report` output (to stderr):

```
  duplicate game #312 (Leaf_vA vs Leaf_vB) in run2.pgn — matches game #87 in run1.pgn

pgn_dedup: 4500 games read, 4487 written, 13 duplicates removed.
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `pgn` | *(required)* | PGN file(s) to process |
| `--output FILE` | stdout | Write deduplicated games to FILE |
| `--report` | off | Print per-duplicate details and a summary to stderr |
| `--players` | off | Include White and Black headers in the identity key |

By default only the move sequence is used for comparison, so games that differ
only in headers (date, round, event) are treated as duplicates.  With `--players`,
two games must also have matching `White` and `Black` tags to be considered duplicates
— useful when the same position was played by different engine pairs.

---

## tdleaf_selfplay.py

*(Not in active use.)*  Earlier self-play driver for TDLeaf training, predating
`training_run.py`.  Kept in `scripts/` for reference; no symlinks are provided
in `run/` or `learn/`.
