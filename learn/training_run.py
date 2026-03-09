#!/usr/bin/env python3
#
# EXchess TDLeaf(λ) training run manager.
# Run from any directory; paths are resolved relative to this script.
#
#   python3 training_run.py
#
# What this script does:
#   1. Asks for a starting .nnue file or offers to initialise a fresh random one.
#   2. Builds a training binary and a read-only binary for the chosen network
#      (via comp.pl in run/), then moves both binaries to learn/ so all
#      training files live together.
#   3. Checks for an existing .tdleaf.bin (continue or start fresh).
#   4. Asks for match parameters, then runs match.py for N games × I iterations.
#      PGN files land in  learn/pgn/<net_base>/  so all PGNs for a given net
#      accumulate in one place across multiple training runs.
#   5. After matches complete, exports the trained weights to a .nnue file whose
#      name encodes the total number of training games to that point:
#        <net_base>-<total_games>g.nnue
#      Game counts are tracked in a sidecar file (learn/<net_base>.games) that
#      persists across multiple runs on the same network.
#
# All working files (.nnue, .tdleaf.bin, .games, binaries, output .nnue) are
# kept in learn/ for easy access.  run/ is only used for comp.pl and match.py.
#

import datetime
import os
import shutil
import subprocess
import sys

learn_dir = os.path.dirname(os.path.abspath(__file__))
run_dir   = os.path.normpath(os.path.join(learn_dir, "../run"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask(prompt, default=None):
    """Prompt the user; return input stripped, or default if empty."""
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (str(default) if default is not None else "")


def ask_yes_no(prompt, default="y"):
    hint = "Y/n" if default == "y" else "y/N"
    val = input(f"{prompt} [{hint}]: ").strip().lower()
    if not val:
        return default == "y"
    return val.startswith("y")


def read_game_count(sidecar_path):
    try:
        with open(sidecar_path) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def write_game_count(sidecar_path, count):
    with open(sidecar_path, "w") as f:
        f.write(f"{count}\n")


def build_binary(version, flags):
    """Invoke comp.pl from run_dir, then move the resulting binary to learn_dir."""
    cmd = ["perl", "comp.pl", version] + flags + ["OVERWRITE"]
    print(f"  $ perl comp.pl {' '.join([version] + flags)} OVERWRITE")
    result = subprocess.run(cmd, cwd=run_dir)
    if result.returncode != 0:
        return False
    built = os.path.join(run_dir, f"EXchess_v{version}")
    dest  = os.path.join(learn_dir, f"EXchess_v{version}")
    if os.path.isfile(built):
        shutil.move(built, dest)
    return os.path.isfile(dest)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cpu_count           = os.cpu_count() or 1
    default_concurrency = max(1, cpu_count // 2)
    date_str            = datetime.datetime.now().strftime("%y%m%d")   # YYMMDD

    print()
    print("=" * 62)
    print("  EXchess TDLeaf(λ) Training Run")
    print("=" * 62)

    # -----------------------------------------------------------------------
    # Step 1 — Net selection
    # -----------------------------------------------------------------------
    print()
    print("Starting network:")
    print("  [1] Use existing .nnue file")
    print("  [2] Initialise a fresh random network")
    choice = ask("Choice", "1")

    if choice.strip() == "2":
        default_fresh = f"nn-fresh-{date_str}.nnue"
        fresh_name    = ask("Output filename for fresh network", default_fresh)
        if not fresh_name.endswith(".nnue"):
            fresh_name += ".nnue"
        net_filename   = fresh_name
        net_base       = os.path.splitext(net_filename)[0]
        net_file       = os.path.join(learn_dir, net_filename)
        do_random_init = True
    else:
        while True:
            net_path = ask("Path to .nnue file").strip()
            if os.path.isfile(net_path):
                break
            print(f"  File not found: {net_path}")
        net_filename   = os.path.basename(net_path)
        net_base       = os.path.splitext(net_filename)[0]
        net_file       = os.path.join(learn_dir, net_filename)
        do_random_init = False
        if os.path.abspath(net_path) != os.path.abspath(net_file):
            print(f"  Copying {net_path} → learn/{net_filename}")
            shutil.copy2(net_path, net_file)

    tdleaf_bin   = os.path.join(learn_dir, net_base + ".tdleaf.bin")
    sidecar_path = os.path.join(learn_dir, net_base + ".games")

    # -----------------------------------------------------------------------
    # Step 2 — Build executables  (comp.pl runs in run/, binaries moved to learn/)
    # -----------------------------------------------------------------------
    print()
    print("Building executables:")
    train_ver = f"train_{net_base}"
    ro_ver    = f"train_{net_base}_ro"
    train_exe = os.path.join(learn_dir, f"EXchess_v{train_ver}")
    ro_exe    = os.path.join(learn_dir, f"EXchess_v{ro_ver}")
    nnue_flag = f"NNUE_NET={net_filename}"

    rebuild = True
    if os.path.isfile(train_exe) and os.path.isfile(ro_exe):
        rebuild = ask_yes_no("  Both executables already exist.  Rebuild?", default="n")

    if rebuild:
        print(f"  Building training binary  (EXchess_v{train_ver}) ...")
        if not build_binary(train_ver, ["NNUE=1", "TDLEAF=1", nnue_flag]):
            print("  Build failed.", file=sys.stderr)
            sys.exit(1)
        print(f"  Building read-only binary (EXchess_v{ro_ver}) ...")
        if not build_binary(ro_ver, ["NNUE=1", "TDLEAF=1", "TDLEAF_READONLY=1", nnue_flag]):
            print("  Build failed.", file=sys.stderr)
            sys.exit(1)
        print("  Build complete.")
    else:
        print("  Using existing binaries.")

    # -----------------------------------------------------------------------
    # Step 2b — Random init (after training binary exists in learn/)
    # -----------------------------------------------------------------------
    if do_random_init:
        print()
        if os.path.isfile(net_file):
            print(f"  Fresh net already exists: {net_filename}")
            overwrite = ask_yes_no("  Overwrite?", default="n")
        else:
            overwrite = True

        if overwrite:
            print(f"  Initialising fresh network → {net_filename}")
            result = subprocess.run(
                [train_exe, "--init-nnue", "--write-nnue", net_file],
                cwd=learn_dir
            )
            if result.returncode != 0:
                print("  --init-nnue failed.", file=sys.stderr)
                sys.exit(1)
        else:
            print("  Using existing fresh net.")

    # -----------------------------------------------------------------------
    # Step 3 — .tdleaf.bin continuity check
    # -----------------------------------------------------------------------
    prior_games = read_game_count(sidecar_path)
    print()
    if os.path.isfile(tdleaf_bin):
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(tdleaf_bin))
        print(f"Found existing {net_base}.tdleaf.bin")
        print(f"  Last modified:   {mtime.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Games on record: {prior_games:,}")
        print("  [1] Continue training from this file")
        print("  [2] Start fresh (rename existing to .tdleaf.bin.bak)")
        cont = ask("Choice", "1")
        if cont.strip() == "2":
            bak = tdleaf_bin + ".bak"
            print(f"  Renaming to {os.path.basename(bak)}")
            os.rename(tdleaf_bin, bak)
            prior_games = 0
            write_game_count(sidecar_path, 0)
    else:
        print(f"No existing .tdleaf.bin for {net_filename} — starting fresh.")

    # -----------------------------------------------------------------------
    # Step 4 — Match parameters
    # -----------------------------------------------------------------------
    print()
    print("Match parameters:")
    n_games     = int(ask("  Games per iteration [-n]        ", 500))
    n_iters     = int(ask("  Iterations          [-i]        ", 10))
    tc          = ask(    "  Time control        [-tc]       ", "0:03+0.05")
    concurrency = int(ask( "  Concurrency         [-c]        ", default_concurrency))
    wait_ms     = int(ask("  Wait between games  [--wait ms] ", 500))
    fischer     = ask_yes_no("  Fischer Random? [--fischer-random]", default="y")

    total_new   = n_games * n_iters
    total_after = prior_games + total_new

    # -----------------------------------------------------------------------
    # PGN directory  —  learn/pgn/<net_base>/
    # -----------------------------------------------------------------------
    pgn_dir       = os.path.join(learn_dir, "pgn", net_base)
    os.makedirs(pgn_dir, exist_ok=True)
    pgn_base_path = os.path.join(pgn_dir, f"match_{net_base}.pgn")

    # -----------------------------------------------------------------------
    # Confirm
    # -----------------------------------------------------------------------
    # net_base already contains the "nn-" prefix (e.g. "nn-ad9b42354671"),
    # so the output name is simply  <net_base>-<games>g.nnue.
    output_net_name = f"{net_base}-{total_after}g.nnue"
    output_net_path = os.path.join(learn_dir, output_net_name)

    print()
    print("=" * 62)
    print(f"  Net in:           {net_filename}")
    print(f"  Training binary:  EXchess_v{train_ver}")
    print(f"  Read-only binary: EXchess_v{ro_ver}")
    print(f"  Games this run:   {total_new:,}  ({n_iters} iter × {n_games} games)")
    print(f"  Prior games:      {prior_games:,}")
    print(f"  Total after run:  {total_after:,}")
    print(f"  TC: {tc}   Concurrency: {concurrency}   Wait: {wait_ms} ms")
    if fischer:
        print( "  Fischer Random:   yes")
    print(f"  PGN directory:    {pgn_dir}/")
    print(f"  Output net:       {output_net_name}  (learn/)")
    print("=" * 62)

    if not ask_yes_no("Proceed?", default="y"):
        print("Aborted.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Step 5 — Run matches
    # -----------------------------------------------------------------------
    print()
    match_py  = os.path.join(run_dir, "match.py")
    match_cmd = [
        sys.executable, match_py,
        train_exe,          # absolute path — match.py passes these straight to cutechess
        ro_exe,
        "-n", str(n_games),
        "-i", str(n_iters),
        "-tc", tc,
        "-c", str(concurrency),
        "--wait", str(wait_ms),
        "-pgn", pgn_base_path,
    ]
    if fischer:
        match_cmd.append("--fischer-random")

    result = subprocess.run(match_cmd, cwd=run_dir)
    if result.returncode != 0:
        print(f"\nMatch run failed (exit code {result.returncode}).", file=sys.stderr)
        print("Weights from completed iterations are preserved in the .tdleaf.bin.")
        print("Game count sidecar NOT updated (re-run to continue from last checkpoint).")
        sys.exit(result.returncode)

    # Update sidecar only after all iterations complete successfully
    write_game_count(sidecar_path, total_after)

    # -----------------------------------------------------------------------
    # Step 6 — Export trained .nnue
    # -----------------------------------------------------------------------
    print()
    print(f"Exporting trained weights → {output_net_name}")
    result = subprocess.run(
        [train_exe, "--write-nnue", output_net_path],
        cwd=learn_dir
    )
    if result.returncode != 0:
        print("--write-nnue failed.", file=sys.stderr)
        sys.exit(result.returncode)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 62)
    print("  Training run complete.")
    print(f"  Net in:      {net_filename}")
    print(f"  Net out:     {output_net_name}  (learn/)")
    print(f"  Games added: {total_new:,}  ({n_iters} iter × {n_games} games)")
    print(f"  Total games: {total_after:,}")
    print(f"  PGN files:   {pgn_dir}/")
    print(f"  .tdleaf.bin: {tdleaf_bin}")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
