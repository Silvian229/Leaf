#!/usr/bin/env python3
#
# Run a head-to-head match or gauntlet between EXchess executables using cutechess-cli.
# Run from the run directory:
#
#   python3 match.py EXchess_vA EXchess_vB [options]
#   python3 match.py EXchess_vA EXchess_vB EXchess_vC ... [options]  # gauntlet
#
# Examples:
#   python3 match.py EXchess_v2026_03_01 EXchess_vtest
#   python3 match.py EXchess_v2026_03_01 EXchess_vtest -n 200 -c 4 -tc 5+0.05
#   python3 match.py EXchess_v2026_03_01 EXchess_vtest -n 400 --openings ../testing/testsuites/wac.epd
#   python3 match.py EXchess_vtrain EXchess_vtrain_ro -n 500 -i 10 --wait 500
#     (10 sequential 500-game matches; engines restart between iterations so the
#      read-only engine picks up the latest .tdleaf.bin weights each time)
#   python3 match.py EXchess_vnew EXchess_v1 EXchess_v2 EXchess_v3 -n 100 --pgn all.pgn
#     (gauntlet: probe engine vs three opponents; all games appended to all.pgn)
#

import argparse
import math
import os
import re
import subprocess
import sys

run_dir       = os.path.dirname(os.path.abspath(__file__))
cutechess_cli = os.path.normpath(os.path.join(run_dir, "../tools/cutechess-1.4.0/build/cutechess-cli"))


def resolve_exe(name):
    """Return absolute path: join with run_dir unless already absolute."""
    return name if os.path.isabs(name) else os.path.join(run_dir, name)


def elo_from_wdl(w, d, l):
    """
    Compute Elo difference and ±95% CI from W/D/L counts (engine1 perspective).
    Returns (elo, elo_err) or (None, None) if score is degenerate.
    """
    n = w + d + l
    if n == 0:
        return None, None
    score = (w + 0.5 * d) / n
    if score <= 0.0 or score >= 1.0:
        return None, None
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    # Propagate Wald std on score → Elo space
    std   = math.sqrt(score * (1.0 - score) / n)
    denom = math.log(10) * score * (1.0 - score)
    elo_err = 400.0 * std / denom if denom != 0 else float("inf")
    return elo, elo_err


def run_match(cmd):
    """
    Run cutechess-cli, stream its output to stdout line-by-line, and capture the
    final Score and Elo difference lines.

    cutechess-cli output format (engine1 perspective):
      Score of E1 vs E2: W - L - D  [pct]  N
      Elo difference: ELO +/- ERR, LOS: ...

    Returns (w, d, l, elo, elo_err, returncode).
    w/d/l are engine1 wins/draws/losses; elo/elo_err may be None.
    """
    score_re = re.compile(
        r"Score of .+? vs .+?:\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)"
    )
    elo_re = re.compile(
        r"Elo difference:\s*([+-]?\d+(?:\.\d+)?)\s*\+/-\s*(\d+(?:\.\d+)?)"
    )

    w = d = l = 0
    elo = elo_err = None

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        m = score_re.search(line)
        if m:
            # cutechess: "W - L - D" from engine1 perspective
            w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        m = elo_re.search(line)
        if m:
            elo     = float(m.group(1))
            elo_err = float(m.group(2))

    proc.wait()
    return w, d, l, elo, elo_err, proc.returncode


def main():
    cpu_count           = os.cpu_count() or 1
    default_concurrency = max(1, cpu_count // 2)

    parser = argparse.ArgumentParser(
        description="Run a match or gauntlet between EXchess versions via cutechess-cli.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("engine1", help="Probe engine (name in run/ or absolute path)")
    parser.add_argument("opponents", nargs="+",
                        help="Opponent engine(s). More than one → gauntlet mode.")
    parser.add_argument("-n", "--games", type=int, default=100,
                        help="Games per iteration per opponent (default: 100)")
    parser.add_argument("-i", "--iterations", type=int, default=1,
                        help="Sequential iterations per opponent (default: 1); engines "
                             "restart between iterations so a read-only TDLeaf engine "
                             "picks up the latest weights each time")
    parser.add_argument("-c", "--concurrency", type=int, default=default_concurrency,
                        help=f"Simultaneous games (default: {default_concurrency})")
    parser.add_argument("-tc", "--time-control", default="10+0.1",
                        help="Time control: 'moves/time+inc' or 'time+inc' in seconds "
                             "(default: 10+0.1)")
    parser.add_argument("--pgn", default=None, metavar="FILE",
                        help="Persistent PGN: all games from all opponents/iterations are "
                             "appended to this file (cutechess -pgnout FILE append)")
    parser.add_argument("--pgn-out", default=None, metavar="FILE",
                        help="Per-iteration PGN base name "
                             "(default: match_<engine1>_vs_<engine2>.pgn); "
                             "with -i > 1 an iteration number is appended before the extension")
    parser.add_argument("--openings", default=None, metavar="FILE",
                        help="Openings file (.epd or .pgn); randomly ordered")
    parser.add_argument("--fischer-random", action="store_true", default=False,
                        help="Use Chess960 / Fischer Random starting positions")
    parser.add_argument("--ponder", action="store_true", default=False,
                        help="Enable pondering (default: off)")
    parser.add_argument("--wait", type=int, default=0, metavar="MS",
                        help="Milliseconds to wait between games (default: 0)")
    args = parser.parse_args()

    # Validate
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.concurrency > cpu_count:
        print(f"Warning: concurrency {args.concurrency} exceeds CPU count {cpu_count}, clamping.",
              file=sys.stderr)
        args.concurrency = cpu_count
    if args.games < 1:
        parser.error("--games must be at least 1")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    # Resolve executables
    exe1 = resolve_exe(args.engine1)
    if not os.path.isfile(exe1):
        print(f"Error: executable not found: {exe1}", file=sys.stderr)
        sys.exit(1)

    opponent_exes = []
    for opp in args.opponents:
        exe = resolve_exe(opp)
        if not os.path.isfile(exe):
            print(f"Error: executable not found: {exe}", file=sys.stderr)
            sys.exit(1)
        opponent_exes.append(exe)

    # Openings
    openings_args = []
    if args.openings:
        if not os.path.isfile(args.openings):
            print(f"Error: openings file not found: {args.openings}", file=sys.stderr)
            sys.exit(1)
        fmt = "epd" if args.openings.lower().endswith(".epd") else "pgn"
        openings_args = ["-openings", f"file={args.openings}", f"format={fmt}", "order=random"]

    name1    = os.path.basename(args.engine1)
    gauntlet = len(args.opponents) > 1
    multi    = args.iterations > 1

    # Header
    print(f"Probe engine: {name1}")
    if gauntlet:
        print(f"Gauntlet vs:  {', '.join(os.path.basename(o) for o in args.opponents)}")
    print(f"Games: {args.games}   Iterations: {args.iterations}   "
          f"Concurrency: {args.concurrency}   TC: {args.time_control}   "
          f"Fischer Random: {'on' if args.fischer_random else 'off'}")
    if args.pgn:
        print(f"Persistent PGN: {args.pgn}")
    if args.openings:
        print(f"Openings:     {args.openings}")
    print()

    gauntlet_results = []   # (name2, total_w, total_d, total_l, display_elo, display_elo_err)
    grand_w = grand_d = grand_l = 0

    for opp_exe, opp_arg in zip(opponent_exes, args.opponents):
        name2    = os.path.basename(opp_arg)
        pgn_base = args.pgn_out or f"match_{name1}_vs_{name2}.pgn"

        # rounds/games setup
        if args.games % 2 == 0:
            rounds_arg = str(args.games // 2)
            games_arg  = ["-games", "2", "-repeat"]
        else:
            rounds_arg = str(args.games)
            games_arg  = []

        base_cmd = [
            cutechess_cli,
            "-engine", f"cmd={exe1}",    f"name={name1}", "proto=xboard", f"dir={run_dir}",
            "-engine", f"cmd={opp_exe}", f"name={name2}", "proto=xboard", f"dir={run_dir}",
            "-each",   f"tc={args.time_control}", *(["ponder"] if args.ponder else []),
            *([ "-variant", "fischerandom"] if args.fischer_random else []),
            "-concurrency", str(args.concurrency),
            "-rounds", rounds_arg,
            "-recover",
            "-draw",   "movenumber=40", "movecount=8", "score=10",
            "-resign", "movecount=6",   "score=600",
            "-ratinginterval", "10",
        ] + games_arg + (["-wait", str(args.wait)] if args.wait > 0 else [])

        if gauntlet:
            print("=" * 60)
            print(f"  vs {name2}")
            print("=" * 60)

        opp_w = opp_d = opp_l = 0
        last_elo = last_elo_err = None

        for it in range(1, args.iterations + 1):
            if args.pgn:
                pgnout_args = ["-pgnout", args.pgn]
                if multi:
                    print(f"--- Iteration {it} / {args.iterations} ---")
                elif not gauntlet:
                    print(f"PGN output:  {args.pgn}  (append)")
            else:
                if multi:
                    root, ext = os.path.splitext(pgn_base)
                    pgn_out   = f"{root}_iter{it:02d}{ext}"
                    print(f"--- Iteration {it} / {args.iterations}   PGN: {pgn_out} ---")
                else:
                    pgn_out = pgn_base
                    if not gauntlet:
                        print(f"PGN output:  {pgn_out}")
                pgnout_args = ["-pgnout", pgn_out]

            cmd = base_cmd + pgnout_args + openings_args

            w, d, l, elo, elo_err, rc = run_match(cmd)

            if rc != 0:
                print(f"\nError: cutechess-cli exited with code {rc} on iteration {it}.",
                      file=sys.stderr)
                sys.exit(rc)

            opp_w += w
            opp_d += d
            opp_l += l
            if elo is not None:
                last_elo, last_elo_err = elo, elo_err

            if multi and it < args.iterations:
                print()

        # Elo for this opponent: aggregate W/D/L when multi-iteration, else cutechess's value
        if multi or last_elo is None:
            display_elo, display_elo_err = elo_from_wdl(opp_w, opp_d, opp_l)
        else:
            display_elo, display_elo_err = last_elo, last_elo_err

        gauntlet_results.append((name2, opp_w, opp_d, opp_l, display_elo, display_elo_err))
        grand_w += opp_w
        grand_d += opp_d
        grand_l += opp_l

        if multi:
            n   = opp_w + opp_d + opp_l
            pct = (opp_w + 0.5 * opp_d) / n * 100 if n else 0
            elo_s = (f"{display_elo:+.0f} ±{display_elo_err:.0f}"
                     if display_elo is not None else "n/a")
            print(f"\nIterations done: {args.iterations} × {args.games} = {n} games  "
                  f"W={opp_w} D={opp_d} L={opp_l} [{pct:.1f}%]  Elo {elo_s}")
        print()

    # Gauntlet summary table
    if gauntlet:
        col_w = 32
        sep   = "=" * 76
        dash  = "-" * 74
        print(sep)
        print(f"  GAUNTLET SUMMARY — {name1}")
        print(sep)
        print(f"  {'Opponent':<{col_w}} {'Games':>5} {'W':>4} {'D':>4} {'L':>4}  "
              f"{'Score%':>7}  {'Elo diff':>10}")
        print("  " + dash)
        for name2, w, d, l, elo, elo_err in gauntlet_results:
            n   = w + d + l
            pct = (w + 0.5 * d) / n * 100 if n else 0
            elo_s = (f"{elo:+.0f} ±{elo_err:.0f}" if elo is not None else "n/a")
            print(f"  {name2:<{col_w}} {n:>5} {w:>4} {d:>4} {l:>4}  {pct:>6.1f}%  {elo_s:>10}")
        print("  " + dash)
        n_tot   = grand_w + grand_d + grand_l
        pct_tot = (grand_w + 0.5 * grand_d) / n_tot * 100 if n_tot else 0
        print(f"  {'TOTAL':<{col_w}} {n_tot:>5} {grand_w:>4} {grand_d:>4} {grand_l:>4}  "
              f"{pct_tot:>6.1f}%  {'':>10}")
        print(sep)

    elif multi:
        print(f"All {args.iterations} iterations complete "
              f"({args.iterations * args.games} games total).")


if __name__ == "__main__":
    main()
