#!/usr/bin/env python3
"""
tdleaf_selfplay.py — Drive a TDLEAF-enabled EXchess binary in self-play
or versus a fixed reference engine.

Single-engine self-play (both sides same binary):
    python3 tdleaf_selfplay.py EXchess_v2026_03_07g_tdleaf -n 200 --depth 6

Two-engine mode (training engine vs fixed opponent):
    python3 tdleaf_selfplay.py EXchess_v2026_03_07g_tdleaf -n 200 --depth 8 \\
        --engine2 EXchess_vtest --depth2 6
    python3 tdleaf_selfplay.py EXchess_v2026_03_07g_tdleaf -n 200 --depth 8 \\
        --engine2 EXchess_v2026_03_07a --tc2 1

Engine 1 must be built with NNUE=1 TDLEAF=1.  Engine 2 can be any EXchess binary.
Scores and wins/losses/draws are always reported from Engine 1's perspective.
In two-engine mode Engine 1 alternates colors each game so TDLeaf accumulates
gradients from both White-to-move and Black-to-move positions.

Protocol (xboard):
    Per game:  new → st <tc> [→ sd <depth>] → go (white engine only)
    Per move:  on "move <uci>" from active engine → send usermove to the other
               (EXchess auto-searches after usermove; no explicit 'go' needed)
    Game end:  result string detected → send "result" to engine 1 → wait for TDLeaf

Time/depth flags:
    --tc SECS     time per move for engine 1  (default: 1.0)
    --depth N     max search depth for engine 1 (overrides --tc when set)
    --tc2 SECS    time per move for engine 2  (default: same as --tc)
    --depth2 N    max search depth for engine 2 (default: same as --depth)
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from queue import Queue, Empty

MOVE_RE   = re.compile(r'\bmove\s+(\S+)$')
RESULT_RE = re.compile(r'\b(1-0|0-1|1/2-1/2)\b')
TDLEAF_OK = re.compile(r'^TDLeaf: updated weights for (\d+)-ply game')
TDLEAF_SK = re.compile(r'^TDLeaf: skipping short game \((\d+) plies\)')


def reader_thread(stream, q, label):
    try:
        for line in stream:
            q.put((label, line.rstrip('\n')))
    except Exception:
        pass
    finally:
        q.put((label, None))


def send(proc, cmd, verbose=False, tag='>>>'):
    if verbose:
        print(f'  {tag} {cmd}', flush=True)
    proc.stdin.write(cmd + '\n')
    proc.stdin.flush()


def drain(q, timeout=0.15):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            q.get(timeout=min(remaining, 0.05))
        except Empty:
            break


def launch_engine(binary, work_dir):
    return subprocess.Popen(
        [binary, 'xb'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=work_dir,
        text=True,
        bufsize=1,
    )


def check_tdleaf_line(line):
    m = TDLEAF_OK.match(line)
    if m:
        return int(m.group(1)), False
    m = TDLEAF_SK.match(line)
    if m:
        return int(m.group(1)), True
    return None


def wait_for_tdleaf(q, timeout, verbose=False, stderr_buf=None):
    """Wait up to *timeout* seconds for a TDLeaf confirmation on e1err.

    Checks *stderr_buf* (lines buffered during gameplay) first, then blocks
    on the shared queue watching for 'e1err' lines.
    """
    for line in (stderr_buf or []):
        result = check_tdleaf_line(line)
        if result:
            return result

    print(f'  [waiting for TDLeaf — up to {timeout:.0f}s]', flush=True)
    deadline = time.monotonic() + timeout
    found    = False
    plies    = None
    skipped  = False

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            label, line = q.get(timeout=min(remaining, 0.5))
        except Empty:
            if found:
                break
            continue

        if line is None:
            break

        if verbose:
            tag = {'e1out': 'E1OUT', 'e1err': 'E1ERR',
                   'e2out': 'E2OUT', 'e2err': 'E2ERR'}.get(label, label.upper())
            print(f'  [{tag}] {line}', flush=True)

        if label == 'e1err':
            result = check_tdleaf_line(line)
            if result:
                plies, skipped = result
                found = True
                deadline = min(deadline, time.monotonic() + 0.25)

    return plies, skipped


def main():
    run_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description='TDLeaf training loop: single-engine self-play or vs reference engine.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('binary',
                        help='TDLeaf training engine (engine 1) name or path')
    parser.add_argument('--engine2', default=None, metavar='BINARY',
                        help='Reference/opponent engine (engine 2). '
                             'Omit for single-engine self-play.')
    parser.add_argument('-n', '--games', type=int, default=100,
                        help='Number of games (default: 100)')
    parser.add_argument('--tc',    type=float, default=1.0,
                        help='Engine 1 time per move in seconds (default: 1.0)')
    parser.add_argument('--tc2',   type=float, default=None,
                        help='Engine 2 time per move in seconds (default: same as --tc)')
    parser.add_argument('--depth',  type=int, default=None,
                        help='Engine 1 max search depth (overrides --tc when set)')
    parser.add_argument('--depth2', type=int, default=None,
                        help='Engine 2 max search depth (default: same as --depth)')
    parser.add_argument('--tdleaf-timeout', type=float, default=15.0,
                        help='Seconds to wait for TDLeaf confirmation per game (default: 15)')
    parser.add_argument('--move-timeout', type=float, default=60.0,
                        help='Seconds to wait for a single move before aborting (default: 60)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print all raw engine I/O')
    args = parser.parse_args()

    def resolve(name):
        if os.path.isabs(name):
            return name
        return os.path.join(run_dir, name)

    binary1 = resolve(args.binary)
    if not os.path.isfile(binary1):
        print(f'Error: binary not found: {binary1}', file=sys.stderr)
        sys.exit(1)

    two_engine = args.engine2 is not None
    if two_engine:
        binary2 = resolve(args.engine2)
        if not os.path.isfile(binary2):
            print(f'Error: engine2 not found: {binary2}', file=sys.stderr)
            sys.exit(1)

    # Defaults for engine 2
    tc2    = args.tc2    if args.tc2    is not None else args.tc
    depth2 = args.depth2 if args.depth2 is not None else args.depth

    work_dir     = os.path.dirname(binary1)
    weights_file = os.path.join(work_dir, 'nn-ad9b42354671.tdleaf.bin')

    # ----------------------------------------------------------------
    # Print configuration
    # ----------------------------------------------------------------
    tc1_str = f'{args.tc}s/move' + (f'  depth≤{args.depth}' if args.depth else '')
    print(f'Mode:         {"two-engine" if two_engine else "single-engine self-play"}')
    print(f'Engine 1:     {os.path.basename(binary1)}')
    print(f'  TC/depth:   {tc1_str}')
    if two_engine:
        tc2_str = f'{tc2}s/move' + (f'  depth≤{depth2}' if depth2 else '')
        print(f'Engine 2:     {os.path.basename(binary2)}')
        print(f'  TC/depth:   {tc2_str}')
    print(f'Games:        {args.games}')
    print(f'Move timeout: {args.move_timeout:.0f}s   TDLeaf timeout: {args.tdleaf_timeout:.0f}s')
    print(f'Work dir:     {work_dir}')
    if args.verbose:
        print('Verbose:      on')
    print()

    # ----------------------------------------------------------------
    # Launch engine(s)
    # ----------------------------------------------------------------
    e1 = launch_engine(binary1, work_dir)
    e2 = launch_engine(binary2, work_dir) if two_engine else None

    q = Queue()
    threading.Thread(target=reader_thread, args=(e1.stdout, q, 'e1out'), daemon=True).start()
    threading.Thread(target=reader_thread, args=(e1.stderr, q, 'e1err'), daemon=True).start()
    if two_engine:
        threading.Thread(target=reader_thread, args=(e2.stdout, q, 'e2out'), daemon=True).start()
        threading.Thread(target=reader_thread, args=(e2.stderr, q, 'e2err'), daemon=True).start()

    # Drain startup banners
    deadline = time.monotonic() + 0.7
    while time.monotonic() < deadline:
        try:
            label, line = q.get(timeout=0.05)
            if args.verbose and line is not None:
                tag = {'e1out': 'E1OUT', 'e1err': 'E1ERR',
                       'e2out': 'E2OUT', 'e2err': 'E2ERR'}.get(label, label.upper())
                print(f'  [STARTUP {tag}] {line}', flush=True)
        except Empty:
            pass

    send(e1, 'xboard', args.verbose, 'E1>>>')
    send(e1, 'easy', args.verbose, 'E1>>>')        # turn off pondering
    send(e1, 'nopost',  args.verbose, 'E1>>>')
    if two_engine:
        send(e2, 'xboard', args.verbose, 'E2>>>')
        send(e2, 'easy', args.verbose, 'E2>>>')    # turn off pondering
        send(e2, 'nopost',  args.verbose, 'E2>>>')
    drain(q, timeout=0.2)

    # ----------------------------------------------------------------
    # Game loop
    # ----------------------------------------------------------------
    wins = draws = losses = 0   # always from engine 1's perspective
    td_plies_list = []
    start_wall    = time.monotonic()
    tc1_int = min(max(1, round(args.tc)),9)
    tc2_int = min(max(1, round(tc2)),9)

    for game_num in range(1, args.games + 1):

        elapsed_so_far = time.monotonic() - start_wall
        print(
            f'--- Game {game_num}/{args.games}'
            f'  (W={wins} D={draws} L={losses}'
            f'  elapsed={elapsed_so_far:.0f}s) ---',
            flush=True,
        )

        # In two-engine mode: alternate engine 1's color each game so TDLeaf
        # sees both White-to-move and Black-to-move positions.
        # Odd games → engine 1 plays White; even games → engine 1 plays Black.
        eng1_white = (not two_engine) or (game_num % 2 == 1)

        # Setup engine 1
        send(e1, 'new',          args.verbose, 'E1>>>')
        send(e1, f'level 0 0:0{tc1_int} 0.05', args.verbose, 'E1>>>')
        if args.depth:
            send(e1, f'sd {args.depth}', args.verbose, 'E1>>>')

        # Setup engine 2 (if present)
        if two_engine:
            send(e2, 'new',          args.verbose, 'E2>>>')
            send(e2, f'level 0 0:0{tc2_int} 0.05', args.verbose, 'E2>>>')
            if depth2:
                send(e2, f'sd {depth2}', args.verbose, 'E2>>>')

        # Kick off: white engine gets 'go'; in single-engine mode that's always e1.
        if two_engine:
            if eng1_white:
                send(e1, 'go', args.verbose, 'E1>>>')
                active,  active_tag  = e1, 'E1>>>'
                passive, passive_tag = e2, 'E2>>>'
                active_lbl = 'e1out'
            else:
                send(e2, 'go', args.verbose, 'E2>>>')
                active,  active_tag  = e2, 'E2>>>'
                passive, passive_tag = e1, 'E1>>>'
                active_lbl = 'e2out'
        else:
            send(e1, 'go', args.verbose, 'E1>>>')

        # ---- play one game ----
        game_result   = None
        move_count    = 0
        game_start    = time.monotonic()
        last_progress = game_start
        e1_stderr_buf = []   # TDLeaf lines may arrive before stdout result

        while game_result is None:
            try:
                label, line = q.get(timeout=args.move_timeout)
            except Empty:
                print(
                    f'  TIMEOUT ({args.move_timeout:.0f}s) waiting for move'
                    f' after {move_count} moves — aborting game.',
                    flush=True,
                )
                game_result = 'timeout'
                break

            if line is None:
                print(f'  Engine stream closed after {move_count} moves.', flush=True)
                game_result = 'error'
                break

            if args.verbose:
                tag = {'e1out': 'E1OUT', 'e1err': 'E1ERR',
                       'e2out': 'E2OUT', 'e2err': 'E2ERR'}.get(label, label.upper())
                print(f'  [{tag}] {line}', flush=True)

            # Buffer engine 1 stderr for TDLeaf detection
            if label == 'e1err':
                e1_stderr_buf.append(line)
                continue
            if label == 'e2err':
                continue

            # ---- stdout from engines ----
            # A move from the active engine triggers usermove+go to the passive one.
            # A result string from either engine ends the game.
            m_move = MOVE_RE.search(line)

            if m_move and (not two_engine or label == active_lbl):
                move = m_move.group(1)
                move_count += 1

                now = time.monotonic()
                if move_count % 10 == 0 or (now - last_progress) >= 5.0:
                    print(f'  move {move_count:3d}  elapsed={now - game_start:.1f}s',
                          flush=True)
                    last_progress = now

                if two_engine:
                    # Inform the passive engine of the move.  EXchess auto-searches
                    # after usermove because p_side!=wtm — sending 'go' here would
                    # trigger a spurious second search with the wrong color.
                    send(passive, f'usermove {move}', args.verbose, passive_tag)
                    # Swap roles.
                    active,  passive  = passive,  active
                    active_tag, passive_tag = passive_tag, active_tag
                    active_lbl = 'e1out' if active is e1 else 'e2out'
                else:
                    # Single-engine: just send go for the other side.
                    send(e1, 'go', args.verbose, 'E1>>>')
                continue

            # Result string from any engine stdout ends the game.
            if label in ('e1out', 'e2out'):
                m_res = RESULT_RE.search(line)
                if m_res:
                    game_result = m_res.group(1)
                    # Don't break yet: in two-engine mode we still need to send
                    # the result to engine 1 to trigger TDLeaf; in single-engine
                    # mode the engine still needs to process the trailing 'go'.

        # ---- post-game ----
        if game_result in ('timeout', 'error'):
            drain(q, timeout=1.0)
            continue

        # In two-engine mode, send the result to engine 1 so TDLeaf is
        # triggered even when engine 2 made the last move.
        if two_engine:
            send(e1, f'result {game_result}', args.verbose, 'E1>>>')

        plies, skipped = wait_for_tdleaf(
            q, args.tdleaf_timeout, args.verbose, e1_stderr_buf
        )
        drain(q, timeout=0.1)

        # Tally from engine 1's perspective.
        if game_result == '1-0':
            if eng1_white: wins   += 1
            else:          losses += 1
        elif game_result == '0-1':
            if eng1_white: losses += 1
            else:          wins   += 1
        else:
            draws += 1

        total = wins + draws + losses

        if plies is not None:
            td_plies_list.append(plies)

        if plies is None:
            td_status = 'TDLeaf: no confirmation (check --tdleaf-timeout or binary flags)'
        elif skipped:
            td_status = f'TDLeaf: skipped ({plies} plies)'
        else:
            td_status = f'TDLeaf: updated {plies} plies'

        color1       = 'W' if eng1_white else 'B'
        game_elapsed = time.monotonic() - game_start
        score_pct    = 100.0 * (wins + 0.5 * draws) / max(1, total)
        print(
            f'  Result: {game_result}  eng1={color1}  moves={move_count}'
            f'  time={game_elapsed:.1f}s'
            f'  W={wins} D={draws} L={losses}  score={score_pct:.1f}%  {td_status}',
            flush=True,
        )

    # ----------------------------------------------------------------
    # Shut down
    # ----------------------------------------------------------------
    for proc, tag in [(e1, 'E1>>>'), (e2, 'E2>>>') if e2 else (None, None)]:
        if proc is None:
            continue
        try:
            send(proc, 'quit', args.verbose, tag)
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    elapsed = time.monotonic() - start_wall
    total   = wins + draws + losses
    avg_len = sum(td_plies_list) / len(td_plies_list) if td_plies_list else 0.0

    print()
    print('=' * 60)
    print(f'Finished {args.games} games in {elapsed:.1f}s'
          f'  ({elapsed / max(1, args.games):.1f}s/game)')
    print(f'Engine 1:  W={wins}  D={draws}  L={losses}'
          f'  score={100.0 * (wins + 0.5 * draws) / max(1, total):.1f}%')
    if td_plies_list:
        print(f'Avg ply count (TDLeaf): {avg_len:.1f}')
    print(f'Weights file: {weights_file}')


if __name__ == '__main__':
    main()
