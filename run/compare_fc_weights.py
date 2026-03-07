#!/usr/bin/env python3
"""
compare_fc_weights.py — visually compare FC layers from a .tdleaf.bin file
against the original FC weights in a .nnue file.

Usage (from run/ directory):
    python3 compare_fc_weights.py nn-ad9b42354671.nnue nn-ad9b42354671.tdleaf.bin
    python3 compare_fc_weights.py nn-ad9b42354671.nnue nn-ad9b42354671.tdleaf.bin --save
"""

import argparse
import os
import struct
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Architecture constants  (nn-ad9b42354671.nnue / Stockfish 15.1 exact)
# ---------------------------------------------------------------------------
L0_SIZE   = 16    # FC0 outputs (15 active + 1 passthrough)
L0_INPUT  = 1024  # FC0 inputs  (SqrCReLU only: 512 per perspective × 2)
L1_SIZE   = 32    # FC1 outputs
L1_PADDED = 32    # FC1 inputs (padded from 30)
L2_PADDED = 32    # FC2 inputs (= FC1 outputs)
N_STACKS  = 8     # layer-stack count
HALF_DIMS = 1024  # FT accumulator width per perspective
FT_INPUTS = 22528 # HalfKAv2_hm feature count
PSQT_BKTS = 8     # PSQT buckets

# Byte sizes
FT_SECTION = (HALF_DIMS * 2 +               # FT biases  (int16)
              FT_INPUTS * HALF_DIMS * 2 +    # FT weights (int16)
              FT_INPUTS * PSQT_BKTS * 4)     # FT PSQT   (int32)

STACK_BYTES = (4 +                           # stack hash
               L0_SIZE * 4 +                 # FC0 biases (int32)
               L0_SIZE * L0_INPUT +          # FC0 weights (int8)
               L1_SIZE * 4 +                 # FC1 biases (int32)
               L1_SIZE * L1_PADDED +         # FC1 weights (int8)
               4 +                           # FC2 bias   (int32)
               L2_PADDED)                    # FC2 weights (int8)

TDLEAF_MAGIC    = 0x544D4C46   # "TMLF"
TDLEAF_VERSION1 = 1
TDLEAF_VERSION2 = 2
TDLEAF_SCALE    = 128.0        # v2: file stores w_f32 × TDLEAF_SCALE

# ---------------------------------------------------------------------------
# vdotq layout converters  (flat vdotq array → natural [o, i] array)
# ---------------------------------------------------------------------------

def _vdotq_to_natural(vdotq_flat, n_out, n_in, stride):
    """
    Convert a flat vdotq-layout weight array to output-major natural layout.

    vdotq layout:  index = (i//4)*stride + (o//4)*16 + (o%4)*4 + (i%4)
    natural layout: index = o * n_in + i

    stride = 64  for FC0 (4 output blocks × 16 bytes each)
    stride = 128 for FC1 (8 output blocks × 16 bytes each)
    """
    o_idx = np.arange(n_out)
    i_idx = np.arange(n_in)
    O, I = np.meshgrid(o_idx, i_idx, indexing='ij')   # [n_out, n_in]
    vdotq_idx = (I // 4) * stride + (O // 4) * 16 + (O % 4) * 4 + (I % 4)
    return vdotq_flat[vdotq_idx]   # shape [n_out, n_in]


def vdotq_to_natural_fc0(flat):
    return _vdotq_to_natural(flat, L0_SIZE, L0_INPUT, 64)


def vdotq_to_natural_fc1(flat):
    return _vdotq_to_natural(flat, L1_SIZE, L1_PADDED, 128)

# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def read_nnue_fc(path):
    """
    Read FC layers from a .nnue file.  Weights are in output-major natural
    layout as stored in the file.  Returns dict of lists (one entry per stack).
    """
    file_size = os.path.getsize(path)
    header_size = file_size - FT_SECTION - N_STACKS * STACK_BYTES
    if header_size < 0:
        sys.exit(f"Error: {path} too small — not a valid .nnue file?")

    data = {k: [] for k in ('fc0_bias', 'fc0_w', 'fc1_bias', 'fc1_w', 'fc2_bias', 'fc2_w')}
    with open(path, 'rb') as f:
        f.seek(header_size + FT_SECTION)
        for _ in range(N_STACKS):
            f.read(4)   # skip stack hash
            data['fc0_bias'].append(np.frombuffer(f.read(L0_SIZE * 4),       dtype=np.int32).copy())
            data['fc0_w'   ].append(np.frombuffer(f.read(L0_SIZE * L0_INPUT),dtype=np.int8 ).reshape(L0_SIZE, L0_INPUT).copy())
            data['fc1_bias'].append(np.frombuffer(f.read(L1_SIZE * 4),       dtype=np.int32).copy())
            data['fc1_w'   ].append(np.frombuffer(f.read(L1_SIZE * L1_PADDED),dtype=np.int8).reshape(L1_SIZE, L1_PADDED).copy())
            data['fc2_bias'].append(np.frombuffer(f.read(4),                 dtype=np.int32).copy())
            data['fc2_w'   ].append(np.frombuffer(f.read(L2_PADDED),         dtype=np.int8 ).copy())
    return data


def read_tdleaf_fc(path):
    """
    Read FC layers from a .tdleaf.bin file.

    v1: biases int32, weights int8 in VDOTQ layout → converted to natural layout.
        No counts.
    v2: biases and weights as float32 (stored × TDLEAF_SCALE) in natural layout,
        followed by uint32 update counts per weight/bias.
        Rounded to int8/int32 for comparison with .nnue.
    """
    cnt_keys = ('fc0_bias_cnt', 'fc0_w_cnt', 'fc1_bias_cnt', 'fc1_w_cnt',
                'fc2_bias_cnt', 'fc2_w_cnt')
    data = {k: [] for k in ('fc0_bias', 'fc0_w', 'fc1_bias', 'fc1_w', 'fc2_bias', 'fc2_w')
                            + cnt_keys}
    data['_has_counts'] = False

    with open(path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        if magic != TDLEAF_MAGIC:
            sys.exit(f"Error: bad magic {magic:#010x} in {path}")

        if version == TDLEAF_VERSION2:
            data['_has_counts'] = True
            for _ in range(N_STACKS):
                def rf(n, fh=f):
                    raw = np.frombuffer(fh.read(n * 4), dtype=np.float32).copy()
                    return raw / TDLEAF_SCALE
                def ru(n, fh=f):
                    return np.frombuffer(fh.read(n * 4), dtype=np.uint32).copy()

                b0 = rf(L0_SIZE)
                data['fc0_bias'    ].append(np.round(b0).astype(np.int32))
                data['fc0_bias_cnt'].append(ru(L0_SIZE))

                w0 = rf(L0_SIZE * L0_INPUT).reshape(L0_SIZE, L0_INPUT)
                data['fc0_w'    ].append(np.clip(np.round(w0), -128, 127).astype(np.int8))
                data['fc0_w_cnt'].append(ru(L0_SIZE * L0_INPUT).reshape(L0_SIZE, L0_INPUT))

                b1 = rf(L1_SIZE)
                data['fc1_bias'    ].append(np.round(b1).astype(np.int32))
                data['fc1_bias_cnt'].append(ru(L1_SIZE))

                w1 = rf(L1_SIZE * L1_PADDED).reshape(L1_SIZE, L1_PADDED)
                data['fc1_w'    ].append(np.clip(np.round(w1), -128, 127).astype(np.int8))
                data['fc1_w_cnt'].append(ru(L1_SIZE * L1_PADDED).reshape(L1_SIZE, L1_PADDED))

                b2 = rf(1)
                data['fc2_bias'    ].append(np.round(b2).astype(np.int32))
                data['fc2_bias_cnt'].append(ru(1))

                w2 = rf(L2_PADDED)
                data['fc2_w'    ].append(np.clip(np.round(w2), -128, 127).astype(np.int8))
                data['fc2_w_cnt'].append(ru(L2_PADDED))

        elif version == TDLEAF_VERSION1:
            for _ in range(N_STACKS):
                data['fc0_bias'].append(np.frombuffer(f.read(L0_SIZE * 4),        dtype=np.int32).copy())
                fc0_vdotq = np.frombuffer(f.read(L0_SIZE * L0_INPUT), dtype=np.int8).copy()
                data['fc0_w'   ].append(vdotq_to_natural_fc0(fc0_vdotq))
                data['fc1_bias'].append(np.frombuffer(f.read(L1_SIZE * 4),        dtype=np.int32).copy())
                fc1_vdotq = np.frombuffer(f.read(L1_SIZE * L1_PADDED), dtype=np.int8).copy()
                data['fc1_w'   ].append(vdotq_to_natural_fc1(fc1_vdotq))
                data['fc2_bias'].append(np.frombuffer(f.read(4),                  dtype=np.int32).copy())
                data['fc2_w'   ].append(np.frombuffer(f.read(L2_PADDED),          dtype=np.int8 ).copy())
            # no counts for v1
        else:
            sys.exit(f"Error: unknown .tdleaf.bin version {version} in {path}")

    return data

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def delta_stats(orig, upd):
    """Return dict of statistics for int8 delta arrays."""
    d = upd.astype(np.int32) - orig.astype(np.int32)
    n = d.size
    n_changed = int(np.sum(d != 0))
    return dict(n=n, n_changed=n_changed,
                pct=100 * n_changed / n,
                dmin=int(d.min()), dmax=int(d.max()),
                dmean=float(d.mean()), dstd=float(d.std()),
                delta=d)


def bias_delta_stats(orig, upd):
    """Return statistics for int32 bias delta arrays."""
    d = upd.astype(np.int64) - orig.astype(np.int64)
    n = d.size
    n_changed = int(np.sum(d != 0))
    return dict(n=n, n_changed=n_changed,
                pct=100 * n_changed / n,
                dmin=int(d.min()), dmax=int(d.max()),
                dmean=float(d.mean()), dstd=float(d.std()),
                delta=d)

# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_summary(orig, upd):
    has_counts = upd.get('_has_counts', False)

    print("\n┌─────────────────────────────────────────────────────────────────────────┐")
    print("│              Weight change summary (all stacks combined)               │")
    print("├──────────┬──────────────┬───────────────┬───────────┬─────────────────┤")
    print("│  Layer   │  Changed     │  % Changed    │  Δ range  │  mean ± std     │")
    print("├──────────┼──────────────┼───────────────┼───────────┼─────────────────┤")

    for layer_name, key_w, key_b in [
            ('FC0 wts',  'fc0_w',   'fc0_bias'),
            ('FC1 wts',  'fc1_w',   'fc1_bias'),
            ('FC2 wts',  'fc2_w',   'fc2_bias'),
    ]:
        all_o = np.concatenate(orig[key_w])
        all_u = np.concatenate(upd[key_w])
        s = delta_stats(all_o, all_u)
        print(f"│ {layer_name:<8} │ {s['n_changed']:>6}/{s['n']:<5} │ {s['pct']:>12.2f}% │"
              f" [{s['dmin']:+4d},{s['dmax']:+4d}] │ {s['dmean']:+7.3f} ± {s['dstd']:.3f}  │")

        # biases (int32)
        all_ob = np.concatenate(orig[key_b])
        all_ub = np.concatenate(upd[key_b])
        sb = bias_delta_stats(all_ob, all_ub)
        print(f"│ {layer_name[:-3]+'bis':<8} │ {sb['n_changed']:>6}/{sb['n']:<5} │ {sb['pct']:>12.2f}% │"
              f" [{sb['dmin']:+4d},{sb['dmax']:+4d}] │ {sb['dmean']:+7.3f} ± {sb['dstd']:.3f}  │")

    print("└──────────┴──────────────┴───────────────┴───────────┴─────────────────┘")

    # Per-stack breakdown
    print("\nPer-stack FC1 weight changes:")
    print(f"  {'Stack':<7} {'Changed':>10} {'%':>8} {'Δ min':>8} {'Δ max':>8} {'|Δ| mean':>10}")
    for s in range(N_STACKS):
        st = delta_stats(orig['fc1_w'][s], upd['fc1_w'][s])
        print(f"  Stack {s}  {st['n_changed']:>6}/{st['n']:<4} {st['pct']:>7.2f}%"
              f"  {st['dmin']:>6}  {st['dmax']:>6}  {abs(st['dmean']):>9.3f}")

    # Update count summary (v2 only)
    if has_counts:
        print("\nUpdate count summary (v2 — times each weight was updated across sessions):")
        print(f"  {'Layer':<10} {'Total wts':>10} {'Ever updated':>14} {'Max cnt':>9} {'Mean cnt (>0)':>14}")
        for layer_name, key_w, key_b in [
                ('FC0 wts',  'fc0_w_cnt',   'fc0_bias_cnt'),
                ('FC1 wts',  'fc1_w_cnt',   'fc1_bias_cnt'),
                ('FC2 wts',  'fc2_w_cnt',   'fc2_bias_cnt'),
        ]:
            wc = np.concatenate(upd[key_w]).ravel()
            nz = wc[wc > 0]
            print(f"  {layer_name:<10} {wc.size:>10} {len(nz):>14} {int(wc.max()):>9}"
                  f" {(nz.mean() if len(nz) else 0.0):>14.2f}")
            bc = np.concatenate(upd[key_b]).ravel()
            bnz = bc[bc > 0]
            label = layer_name[:-3] + 'bis'
            print(f"  {label:<10} {bc.size:>10} {len(bnz):>14} {int(bc.max()):>9}"
                  f" {(bnz.mean() if len(bnz) else 0.0):>14.2f}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(orig, upd, save):
    """3×3 figure: one row per FC layer.  Cols: orig dist | delta dist | per-stack bar."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    nnue_name   = getattr(orig,  '_name', 'original .nnue')
    tdleaf_name = getattr(upd,   '_name', 'updated .tdleaf.bin')

    fig.suptitle(f'FC weight comparison\n'
                 f'blue = {nnue_name}   orange = delta (updated − original)',
                 fontsize=11)

    layers = [
        ('FC0 weights (16×1024)',  'fc0_w'),
        ('FC1 weights (32×32)',    'fc1_w'),
        ('FC2 weights (32)',       'fc2_w'),
    ]

    for row, (layer_name, key) in enumerate(layers):
        all_orig = np.concatenate(orig[key]).ravel().astype(np.int32)
        all_upd  = np.concatenate(upd[key]).ravel().astype(np.int32)
        delta    = all_upd - all_orig

        # --- col 0: original weight distribution ---
        ax = axes[row, 0]
        bins = np.arange(-128, 130) - 0.5
        ax.hist(all_orig, bins=bins, color='steelblue', alpha=0.75, density=True)
        ax.set_title(f'{layer_name}\noriginal distribution')
        ax.set_xlabel('weight value')
        ax.set_ylabel('density')
        ax.set_xlim(-130, 130)

        # --- col 1: delta distribution ---
        ax = axes[row, 1]
        d_max = max(abs(int(delta.min())), abs(int(delta.max())), 1)
        d_bins = np.arange(-d_max - 1, d_max + 2) - 0.5
        ax.hist(delta, bins=d_bins, color='coral', alpha=0.85)
        n_nz = int(np.sum(delta != 0))
        ax.set_title(f'{layer_name}\nΔ distribution  ({n_nz}/{delta.size} changed)')
        ax.set_xlabel('Δ value (updated − original)')
        ax.set_ylabel('count')
        ax.axvline(0, color='k', linewidth=0.9, linestyle='--')

        # --- col 2: per-stack changed fraction + max|Δ| ---
        ax = axes[row, 2]
        pcts  = []
        maxds = []
        for s in range(N_STACKS):
            d = upd[key][s].astype(np.int32) - orig[key][s].astype(np.int32)
            pcts.append(100 * np.mean(d != 0))
            maxds.append(int(np.max(np.abs(d))))
        x = np.arange(N_STACKS)
        ax.bar(x, pcts, color='mediumseagreen', alpha=0.80)
        ax2 = ax.twinx()
        ax2.plot(x, maxds, 'r^--', markersize=7, label='max |Δ|')
        ax2.set_ylabel('max |Δ|', color='red', fontsize=9)
        ax2.tick_params(axis='y', colors='red')
        ax2.legend(loc='upper right', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{i}' for i in x])
        ax.set_title(f'{layer_name}\n% changed per stack')
        ax.set_xlabel('stack')
        ax.set_ylabel('% weights changed')

    plt.tight_layout()
    if save:
        fig.savefig('fc_compare_overview.png', dpi=150)
        print("Saved fc_compare_overview.png")
    return fig


def plot_fc1_per_stack(orig, upd, save):
    """2×4 figure: FC1 weight distributions per stack (orig vs updated overlay)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle('FC1 weights — original (blue) vs updated (orange), per stack', fontsize=12)
    bins = np.arange(-128, 130) - 0.5

    for s in range(N_STACKS):
        ax = axes[s // 4][s % 4]
        o = orig['fc1_w'][s].flatten().astype(int)
        u = upd['fc1_w'][s].flatten().astype(int)
        d = u - o
        n_changed = int(np.sum(d != 0))

        ax.hist(o, bins=bins, alpha=0.55, color='steelblue', density=True, label='orig')
        ax.hist(u, bins=bins, alpha=0.55, color='orange',    density=True, label='updated')
        ax.set_title(f'Stack {s}  ({n_changed}/{d.size} changed)', fontsize=10)
        ax.set_xlabel('weight value', fontsize=8)
        ax.set_ylabel('density', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xlim(-130, 130)

    plt.tight_layout()
    if save:
        fig.savefig('fc_compare_fc1_stacks.png', dpi=150)
        print("Saved fc_compare_fc1_stacks.png")
    return fig


def plot_delta_heatmaps(orig, upd, save):
    """Show delta (updated − original) as heatmaps for FC0 and FC1."""
    fig, axes = plt.subplots(2, N_STACKS, figsize=(18, 6))
    fig.suptitle('FC weight deltas (updated − original) — each cell = one weight',
                 fontsize=11)

    for s in range(N_STACKS):
        # FC0: 16 outputs × 1024 inputs — show as 16×1024 image
        ax = axes[0, s]
        d0 = (upd['fc0_w'][s].astype(np.int32) - orig['fc0_w'][s].astype(np.int32))
        vmax = max(abs(int(d0.min())), abs(int(d0.max())), 1)
        im = ax.imshow(d0, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title(f'FC0 S{s}', fontsize=9)
        ax.set_xlabel('input', fontsize=7)
        if s == 0:
            ax.set_ylabel('output', fontsize=8)
        else:
            ax.set_yticks([])

        # FC1: 32 outputs × 32 inputs — show as 32×32 image
        ax = axes[1, s]
        d1 = (upd['fc1_w'][s].astype(np.int32) - orig['fc1_w'][s].astype(np.int32))
        vmax = max(abs(int(d1.min())), abs(int(d1.max())), 1)
        im = ax.imshow(d1, aspect='equal', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title(f'FC1 S{s}', fontsize=9)
        ax.set_xlabel('input', fontsize=7)
        if s == 0:
            ax.set_ylabel('output', fontsize=8)
        else:
            ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save:
        fig.savefig('fc_compare_heatmaps.png', dpi=150)
        print("Saved fc_compare_heatmaps.png")
    return fig


def plot_bias_changes(orig, upd, save):
    """Show FC bias changes (int32) for each layer and stack."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('FC bias changes (updated − original, int32 scale)', fontsize=11)

    for col, (layer_name, key_b) in enumerate([
            ('FC0 biases (16 per stack)',  'fc0_bias'),
            ('FC1 biases (32 per stack)',  'fc1_bias'),
            ('FC2 bias  (1 per stack)',    'fc2_bias'),
    ]):
        ax = axes[col]
        x_off = 0
        xticks, xlabels = [], []
        for s in range(N_STACKS):
            o = orig[key_b][s].astype(np.int64)
            u = upd[key_b][s].astype(np.int64)
            d = u - o
            x = np.arange(len(d)) + x_off
            colors = ['red' if v != 0 else 'steelblue' for v in d]
            ax.bar(x, d, color=colors, alpha=0.8)
            mid = x_off + len(d) / 2
            xticks.append(mid)
            xlabels.append(f'S{s}')
            x_off += len(d) + 1   # gap between stacks

        ax.axhline(0, color='k', linewidth=0.7)
        ax.set_title(layer_name)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_ylabel('Δ bias (int32 units)')
        ax.set_xlabel('stack')

    plt.tight_layout()
    if save:
        fig.savefig('fc_compare_biases.png', dpi=150)
        print("Saved fc_compare_biases.png")
    return fig

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Compare FC weights: original .nnue vs updated .tdleaf.bin',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('nnue',   help='.nnue file (original weights)')
    ap.add_argument('tdleaf', help='.tdleaf.bin file (updated weights)')
    ap.add_argument('--save', action='store_true',
                    help='Save plots to PNG files instead of (or in addition to) showing')
    ap.add_argument('--no-show', action='store_true',
                    help='Do not open interactive windows (implies --save)')
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use('Agg')
        args.save = True
    else:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            pass   # fall back to whatever default is available

    for path in (args.nnue, args.tdleaf):
        if not os.path.isfile(path):
            sys.exit(f"Error: file not found: {path}")

    print(f"Reading {args.nnue} ...")
    orig = read_nnue_fc(args.nnue)
    orig['_name'] = os.path.basename(args.nnue)

    print(f"Reading {args.tdleaf} ...")
    upd = read_tdleaf_fc(args.tdleaf)
    upd['_name'] = os.path.basename(args.tdleaf)

    print_summary(orig, upd)

    plot_overview(orig, upd, args.save)
    plot_fc1_per_stack(orig, upd, args.save)
    plot_delta_heatmaps(orig, upd, args.save)
    plot_bias_changes(orig, upd, args.save)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
