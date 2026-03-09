#!/usr/bin/env python3
#
# Run a head-to-head match between two EXchess executables using cutechess-cli.
# Run from the run directory:
#
#   python3 match.py EXchess_vA EXchess_vB [options]
#
# Examples:
#   python3 match.py EXchess_v2026_03_01 EXchess_vtest
#   python3 match.py EXchess_v2026_03_01 EXchess_vtest -n 200 -c 4 -tc 5+0.05
#   python3 match.py EXchess_v2026_03_01 EXchess_vtest -n 400 --openings ../testing/testsuites/wac.epd
#   python3 match.py EXchess_vtrain EXchess_vtrain_ro -n 500 -i 10 --wait 500
#     (10 sequential 500-game matches; engines restart between iterations so the
#      read-only engine picks up the latest .tdleaf.bin weights each time)
#

import argparse
import os
import subprocess
import sys

run_dir       = os.path.dirname(os.path.abspath(__file__))
cutechess_cli = os.path.normpath(os.path.join(run_dir, "../tools/cutechess-1.4.0/build/cutechess-cli"))

def resolve_exe(name):
    """Return absolute path: join with run_dir unless already absolute."""
    return name if os.path.isabs(name) else os.path.join(run_dir, name)

def main():
    cpu_count = os.cpu_count() or 1
    default_concurrency = max(1, cpu_count // 2)

    parser = argparse.ArgumentParser(
        description="Run a match between two EXchess versions via cutechess-cli.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("engine1", help="First EXchess executable (name in run/ or absolute path)")
    parser.add_argument("engine2", help="Second EXchess executable (name in run/ or absolute path)")
    parser.add_argument("-n", "--games", type=int, default=100,
                        help="Total number of games to play per iteration (default: 100)")
    parser.add_argument("-i", "--iterations", type=int, default=1,
                        help="Number of sequential match iterations (default: 1); engines restart "
                             "between iterations so a read-only TDLeaf engine picks up the latest "
                             "weights at the start of each new match")
    parser.add_argument("-c", "--concurrency", type=int, default=default_concurrency,
                        help=f"Simultaneous games (default: {default_concurrency}, max: {cpu_count})")
    parser.add_argument("-tc", "--time-control", default="10+0.1",
                        help="Time control: 'moves/time+inc' or 'time+inc' in seconds (default: 10+0.1)")
    parser.add_argument("-pgn", "--pgn-out", default=None,
                        help="PGN output file base name (default: match_<engine1>_<engine2>.pgn); "
                             "with -i > 1 an iteration number is appended before the extension")
    parser.add_argument("--openings", default=None, metavar="FILE",
                        help="Openings file (.epd or .pgn); randomly ordered")
    parser.add_argument("--ponder", action="store_true", default=False,
                        help="Enable pondering (default: off)")
    parser.add_argument("--wait", type=int, default=0, metavar="MS",
                        help="Milliseconds to wait between games (-wait in cutechess-cli, default: 0)")
    args = parser.parse_args()

    # Validate concurrency
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.concurrency > cpu_count:
        print(f"Warning: concurrency {args.concurrency} exceeds CPU count {cpu_count}, clamping.",
              file=sys.stderr)
        args.concurrency = cpu_count

    # Validate games and iterations
    if args.games < 1:
        parser.error("--games must be at least 1")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    # Resolve and validate executables
    exe1 = resolve_exe(args.engine1)
    exe2 = resolve_exe(args.engine2)
    for exe in (exe1, exe2):
        if not os.path.isfile(exe):
            print(f"Error: executable not found: {exe}", file=sys.stderr)
            sys.exit(1)

    name1 = os.path.basename(args.engine1)
    name2 = os.path.basename(args.engine2)

    pgn_base = args.pgn_out or f"match_{name1}_vs_{name2}.pgn"

    # Build the cutechess-cli command template (PGN path filled in per iteration).
    # -rounds N plays N games for a two-engine match (one game per round).
    # -games 2 with -rounds N//2 ensures each opening is played from both
    # sides; we use this when the game count is even.
    if args.games % 2 == 0:
        rounds_arg = str(args.games // 2)
        games_arg  = ["-games", "2", "-repeat"]
    else:
        rounds_arg = str(args.games)
        games_arg  = []

    base_cmd = [
        cutechess_cli,
        "-engine", f"cmd={exe1}", f"name={name1}", "proto=xboard", f"dir={run_dir}",
        "-engine", f"cmd={exe2}", f"name={name2}", "proto=xboard", f"dir={run_dir}",
        "-each",   f"tc={args.time_control}", *(["ponder"] if args.ponder else []),
        "-concurrency", str(args.concurrency),
        "-rounds", rounds_arg,
        "-recover",
        "-draw",   "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=6",   "score=600",
        "-ratinginterval", "10",
    ] + games_arg + (["-wait", str(args.wait)] if args.wait > 0 else [])

    openings_args = []
    if args.openings:
        if not os.path.isfile(args.openings):
            print(f"Error: openings file not found: {args.openings}", file=sys.stderr)
            sys.exit(1)
        fmt = "epd" if args.openings.lower().endswith(".epd") else "pgn"
        openings_args = ["-openings", f"file={args.openings}", f"format={fmt}", "order=random"]

    multi = args.iterations > 1

    print(f"Match:       {name1}  vs  {name2}")
    print(f"Games:       {args.games}   Iterations: {args.iterations}   "
          f"Concurrency: {args.concurrency}   TC: {args.time_control}   "
          f"Ponder: {'on' if args.ponder else 'off'}")
    if args.openings:
        print(f"Openings:    {args.openings}")
    print()

    for it in range(1, args.iterations + 1):
        if multi:
            # Insert iteration number before the file extension.
            root, ext = os.path.splitext(pgn_base)
            pgn_out = f"{root}_iter{it:02d}{ext}"
            print(f"--- Iteration {it} / {args.iterations}   PGN: {pgn_out} ---")
        else:
            pgn_out = pgn_base
            print(f"PGN output:  {pgn_out}")

        cmd = base_cmd + ["-pgnout", pgn_out] + openings_args

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nError: cutechess-cli exited with code {result.returncode} on iteration {it}.",
                  file=sys.stderr)
            sys.exit(result.returncode)

        if multi and it < args.iterations:
            print()

    if multi:
        print(f"\nAll {args.iterations} iterations complete "
              f"({args.iterations * args.games} games total).")

if __name__ == "__main__":
    main()
