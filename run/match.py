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
#

import argparse
import os
import subprocess
import sys

run_dir      = os.path.dirname(os.path.abspath(__file__))
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
                        help="Total number of games to play (default: 100)")
    parser.add_argument("-c", "--concurrency", type=int, default=default_concurrency,
                        help=f"Simultaneous games (default: {default_concurrency}, max: {cpu_count})")
    parser.add_argument("-tc", "--time-control", default="10+0.1",
                        help="Time control: 'moves/time+inc' or 'time+inc' in seconds (default: 10+0.1)")
    parser.add_argument("-pgn", "--pgn-out", default=None,
                        help="PGN output file (default: match_<engine1>_<engine2>.pgn)")
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

    # Validate games
    if args.games < 1:
        parser.error("--games must be at least 1")

    # Resolve and validate executables
    exe1 = resolve_exe(args.engine1)
    exe2 = resolve_exe(args.engine2)
    for exe in (exe1, exe2):
        if not os.path.isfile(exe):
            print(f"Error: executable not found: {exe}", file=sys.stderr)
            sys.exit(1)

    name1 = os.path.basename(args.engine1)
    name2 = os.path.basename(args.engine2)

    pgn_out = args.pgn_out or f"match_{name1}_vs_{name2}.pgn"

    # Build cutechess-cli command.
    # -rounds N plays N games for a two-engine match (one game per round).
    # -games 2 with -rounds N//2 ensures each opening is played from both
    # sides; we use this when the game count is even.
    if args.games % 2 == 0:
        rounds_arg = str(args.games // 2)
        games_arg  = ["-games", "2", "-repeat"]
    else:
        rounds_arg = str(args.games)
        games_arg  = []

    cmd = [
        cutechess_cli,
        "-engine", f"cmd={exe1}", f"name={name1}", "proto=xboard", f"dir={run_dir}",
        "-engine", f"cmd={exe2}", f"name={name2}", "proto=xboard", f"dir={run_dir}",
        "-each",   f"tc={args.time_control}", *(["ponder"] if args.ponder else []),
        "-concurrency", str(args.concurrency),
        "-rounds", rounds_arg,
        "-recover",
        "-draw",   "movenumber=40", "movecount=8", "score=10",
        "-resign", "movecount=6",   "score=600",
        "-pgnout", pgn_out,
        "-ratinginterval", "10",
    ] + games_arg + (["-wait", str(args.wait)] if args.wait > 0 else [])

    if args.openings:
        if not os.path.isfile(args.openings):
            print(f"Error: openings file not found: {args.openings}", file=sys.stderr)
            sys.exit(1)
        fmt = "epd" if args.openings.lower().endswith(".epd") else "pgn"
        cmd += ["-openings", f"file={args.openings}", f"format={fmt}", "order=random"]

    print(f"Match:       {name1}  vs  {name2}")
    print(f"Games:       {args.games}   Concurrency: {args.concurrency}   TC: {args.time_control}   Ponder: {'on' if args.ponder else 'off'}")
    print(f"PGN output:  {pgn_out}")
    if args.openings:
        print(f"Openings:    {args.openings}")
    print()

    os.execv(cutechess_cli, cmd)

if __name__ == "__main__":
    main()
