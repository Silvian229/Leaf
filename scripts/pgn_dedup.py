#!/usr/bin/env python3
"""
pgn_dedup.py -- remove duplicate games from PGN files.

Two games are considered identical when their move sequences match after
stripping comments, annotations, and move numbers.  Use --players to also
require matching White/Black headers.

Usage:
    python3 scripts/pgn_dedup.py <pgn_file> [<pgn_file> ...] [options]

Options:
    --output FILE     write deduplicated games to FILE (default: stdout)
    --report          print duplicate-removal summary to stderr
    --players         include White and Black headers in the identity key
"""

import argparse
import hashlib
import re
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Remove duplicate games from one or more PGN files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pgn", nargs="+", help="PGN file(s) to process")
    p.add_argument("--output", default="-",
                   help="output file (default: stdout)")
    p.add_argument("--report", action="store_true",
                   help="print summary of duplicates removed to stderr")
    p.add_argument("--players", action="store_true",
                   help="include White and Black headers in the identity key")
    return p.parse_args()


# Matches a PGN tag pair: [Name "Value"]
_TAG_RE  = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
# Strips move numbers (e.g. "1." "12." "1..." "12..."), comments ({...}),
# NAG annotations ($12), and extra whitespace.
_MOVNUM_RE  = re.compile(r'\d+\.+')
_COMMENT_RE = re.compile(r'\{[^}]*\}')
_NAG_RE     = re.compile(r'\$\d+')
_RESULT_RE  = re.compile(r'(1-0|0-1|1/2-1/2|\*)\s*$')


def normalise_moves(movetext):
    """Return a canonical move string for identity comparison."""
    s = _COMMENT_RE.sub(' ', movetext)
    s = _NAG_RE.sub(' ', s)
    s = _MOVNUM_RE.sub(' ', s)
    s = _RESULT_RE.sub(' ', s)
    return ' '.join(s.split())


def iter_games(path):
    """
    Yield (headers_dict, raw_game_text) for each game in a PGN file.
    raw_game_text is the complete text of the game including tags and movetext.
    """
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError as e:
        sys.exit(f"Error reading {path}: {e}")

    game_lines = []
    headers = {}

    for line in lines:
        stripped = line.rstrip('\n')
        m = _TAG_RE.match(stripped)
        if m:
            headers[m.group(1)] = m.group(2)
            game_lines.append(line)
        elif stripped == '' and game_lines:
            # Blank line: if we already have movetext flush the game,
            # otherwise it's just spacing between tags and movetext.
            movetext = ''.join(
                l for l in game_lines if not _TAG_RE.match(l.rstrip('\n'))
            )
            if movetext.strip():
                yield headers, ''.join(game_lines)
                game_lines = []
                headers = {}
            else:
                game_lines.append(line)
        else:
            game_lines.append(line)

    # Flush final game (file may not end with a blank line)
    if game_lines:
        movetext = ''.join(
            l for l in game_lines if not _TAG_RE.match(l.rstrip('\n'))
        )
        if movetext.strip():
            yield headers, ''.join(game_lines)


def game_key(headers, movetext, use_players):
    """Return a hash that identifies the game for deduplication purposes."""
    norm = normalise_moves(movetext)
    if use_players:
        white = headers.get('White', '')
        black = headers.get('Black', '')
        key_str = f"{white}\x00{black}\x00{norm}"
    else:
        key_str = norm
    return hashlib.md5(key_str.encode()).hexdigest()


def main():
    args = parse_args()

    out = sys.stdout if args.output == '-' else open(args.output, 'w', encoding='utf-8')

    seen = {}       # key -> (source_file, game_index) for reporting
    total = 0
    written = 0
    dupes = 0

    try:
        for path in args.pgn:
            for headers, raw in iter_games(path):
                total += 1
                movetext = ''.join(
                    l for l in raw.splitlines(keepends=True)
                    if not _TAG_RE.match(l.rstrip('\n'))
                )
                key = game_key(headers, movetext, args.players)

                if key in seen:
                    dupes += 1
                    if args.report:
                        orig_path, orig_idx = seen[key]
                        w = headers.get('White', '?')
                        b = headers.get('Black', '?')
                        print(
                            f"  duplicate game #{total} ({w} vs {b}) in {path}"
                            f" — matches game #{orig_idx} in {orig_path}",
                            file=sys.stderr,
                        )
                else:
                    seen[key] = (path, total)
                    out.write(raw)
                    if not raw.endswith('\n\n'):
                        out.write('\n' if raw.endswith('\n') else '\n\n')
                    written += 1
    finally:
        if out is not sys.stdout:
            out.close()

    if args.report:
        print(
            f"\npgn_dedup: {total} games read, {written} written, {dupes} duplicates removed.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
