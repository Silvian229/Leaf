#!/usr/bin/env python3
"""
compare_fc_weights.py — compare FC layers from a .tdleaf.bin file against the
original .nnue weights; also summarise FT and PSQT layers (now trainable by TDLeaf).

Usage (from run/ directory):
    python3 compare_fc_weights.py nn-ad9b42354671.nnue nn-ad9b42354671.tdleaf.bin
    python3 compare_fc_weights.py nn-ad9b42354671.nnue nn-ad9b42354671.tdleaf.bin --save
    python3 compare_fc_weights.py nn-ad9b42354671.nnue nn-ad9b42354671.tdleaf.bin --ft-weights
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
TDLEAF_VERSION3 = 3
TDLEAF_VERSION4 = 4            # adds FT bias section
TDLEAF_SCALE    = 128.0        # v2+: file stores w_f32 × TDLEAF_SCALE

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

# ---------------------------------------------------------------------------
# LEB128 helpers  (used for reading FT biases, weights, and PSQT from .nnue)
# ---------------------------------------------------------------------------

LEB128_MAGIC = b'COMPRESSED_LEB128'

def _decode_leb128_i16(data: bytes, count: int) -> np.ndarray:
    """Decode a signed-LEB128 byte stream into a numpy int16 array."""
    out = np.empty(count, dtype=np.int32)
    pos = 0
    for i in range(count):
        val = shift = 0
        while True:
            b = data[pos]; pos += 1
            val |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        if shift < 16 and (data[pos - 1] & 0x40):
            val |= -(1 << shift)
        out[i] = val
    return out.astype(np.int16)


def _decode_leb128_i32(data: bytes, count: int) -> np.ndarray:
    """Decode a signed-LEB128 byte stream into a numpy int32 array."""
    out = np.empty(count, dtype=np.int64)
    pos = 0
    for i in range(count):
        val = shift = 0
        while True:
            b = data[pos]; pos += 1
            val |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        if shift < 32 and (data[pos - 1] & 0x40):
            val |= -(1 << shift)
        out[i] = val
    return out.astype(np.int32)


def _read_leb128_section_i16(f, count):
    magic = f.read(17)
    if magic != LEB128_MAGIC:
        f.seek(-17, 1)
        return np.frombuffer(f.read(count * 2), dtype='<i2').copy()
    nbytes = struct.unpack('<I', f.read(4))[0]
    return _decode_leb128_i16(f.read(nbytes), count)


def _read_leb128_section_i32(f, count):
    magic = f.read(17)
    if magic != LEB128_MAGIC:
        f.seek(-17, 1)
        return np.frombuffer(f.read(count * 4), dtype='<i4').copy()
    nbytes = struct.unpack('<I', f.read(4))[0]
    return _decode_leb128_i32(f.read(nbytes), count)


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _skip_leb128_or_raw_i16(f, count):
    """Skip a LEB128-compressed or raw int16 section without decoding it."""
    magic = f.read(17)
    if magic == LEB128_MAGIC:
        nbytes = struct.unpack('<I', f.read(4))[0]
        f.seek(nbytes, 1)
    else:
        f.seek(-17 + count * 2, 1)


def _skip_leb128_or_raw_i32(f, count):
    """Skip a LEB128-compressed or raw int32 section without decoding it."""
    magic = f.read(17)
    if magic == LEB128_MAGIC:
        nbytes = struct.unpack('<I', f.read(4))[0]
        f.seek(nbytes, 1)
    else:
        f.seek(-17 + count * 4, 1)


def read_nnue_fc(path):
    """
    Read FC layers from a .nnue file.  Weights are in output-major natural
    layout as stored in the file.  Returns dict of lists (one entry per stack).

    Navigates the header and FT section dynamically so it works for both
    raw and LEB128-compressed .nnue files (e.g. written by nnue_write_nnue).
    """
    data = {k: [] for k in ('fc0_bias', 'fc0_w', 'fc1_bias', 'fc1_w', 'fc2_bias', 'fc2_w')}
    with open(path, 'rb') as f:
        # Parse fixed header
        version_b = f.read(4)
        if len(version_b) < 4:
            sys.exit(f"Error: {path} too small — not a valid .nnue file?")
        f.read(4)                                        # file hash
        desc_size = struct.unpack('<I', f.read(4))[0]
        f.seek(desc_size, 1)                             # skip description
        f.read(4)                                        # ft_hash

        # Skip FT section (LEB128 or raw)
        _skip_leb128_or_raw_i16(f, HALF_DIMS)                    # FT biases
        _skip_leb128_or_raw_i16(f, FT_INPUTS * HALF_DIMS)        # FT weights
        _skip_leb128_or_raw_i32(f, FT_INPUTS * PSQT_BKTS)        # PSQT weights

        # FC stacks
        for _ in range(N_STACKS):
            f.read(4)   # skip stack hash
            data['fc0_bias'].append(np.frombuffer(f.read(L0_SIZE * 4),        dtype=np.int32).copy())
            data['fc0_w'   ].append(np.frombuffer(f.read(L0_SIZE * L0_INPUT), dtype=np.int8 ).reshape(L0_SIZE, L0_INPUT).copy())
            data['fc1_bias'].append(np.frombuffer(f.read(L1_SIZE * 4),        dtype=np.int32).copy())
            data['fc1_w'   ].append(np.frombuffer(f.read(L1_SIZE * L1_PADDED),dtype=np.int8 ).reshape(L1_SIZE, L1_PADDED).copy())
            data['fc2_bias'].append(np.frombuffer(f.read(4),                  dtype=np.int32).copy())
            data['fc2_w'   ].append(np.frombuffer(f.read(L2_PADDED),          dtype=np.int8 ).copy())
    return data


def read_nnue_ft(path, read_ft_weights=False):
    """
    Read FT biases, PSQT weights (and optionally FT weights) from a .nnue file.

    Navigates the header dynamically (desc_size is variable), so it works
    correctly regardless of LEB128 compression in the FT section.

    Returns a dict with:
      'ft_bias'   — int16 [HALF_DIMS]
      'psqt_w'    — int32 [FT_INPUTS × PSQT_BKTS]
      'ft_w'      — int16 [FT_INPUTS × HALF_DIMS]  (only if read_ft_weights=True)
      'ft_w_read' — bool indicating whether ft_w was loaded
    """
    result = {'ft_w_read': False}

    with open(path, 'rb') as f:
        # Parse header to find exact FT section start.
        _version  = f.read(4)
        _filehash = f.read(4)
        desc_size = struct.unpack('<I', f.read(4))[0]
        f.seek(desc_size, 1)          # skip architecture description
        f.read(4)                     # skip ft_hash

        # FT biases: HALF_DIMS int16 (LEB128)
        print("  Reading FT biases ...", flush=True)
        result['ft_bias'] = _read_leb128_section_i16(f, HALF_DIMS)

        # FT weights: FT_INPUTS × HALF_DIMS int16 (LEB128, large)
        ft_w_count = FT_INPUTS * HALF_DIMS  # 23,068,672
        if read_ft_weights:
            print(f"  Reading FT weights ({ft_w_count:,} int16, may take ~30s) ...", flush=True)
            result['ft_w'] = _read_leb128_section_i16(f, ft_w_count)
            result['ft_w_read'] = True
        else:
            # Skip: read the compression header to know how many bytes to jump.
            magic = f.read(17)
            if magic == LEB128_MAGIC:
                nbytes = struct.unpack('<I', f.read(4))[0]
                f.seek(nbytes, 1)              # skip compressed payload
            else:
                f.seek(-17 + ft_w_count * 2, 1)   # skip raw int16 data

        # PSQT weights: FT_INPUTS × PSQT_BKTS int32 (LEB128)
        print("  Reading PSQT weights ...", flush=True)
        psqt_count = FT_INPUTS * PSQT_BKTS   # 180,224
        result['psqt_w'] = _read_leb128_section_i32(f, psqt_count).reshape(FT_INPUTS, PSQT_BKTS)

    return result


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

        if version in (TDLEAF_VERSION2, TDLEAF_VERSION3, TDLEAF_VERSION4):
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

        # v3/v4: sparse FT/PSQT section after FC stacks
        if version in (TDLEAF_VERSION3, TDLEAF_VERSION4):
            n_ft_rows = struct.unpack('<I', f.read(4))[0]
            if n_ft_rows > 0:
                fi_arr   = np.empty(n_ft_rows, dtype=np.uint32)
                ft_w_arr = np.empty((n_ft_rows, HALF_DIMS), dtype=np.float32)
                ft_c_arr = np.empty((n_ft_rows, HALF_DIMS), dtype=np.uint32)
                ps_w_arr = np.empty((n_ft_rows, PSQT_BKTS), dtype=np.float32)
                ps_c_arr = np.empty((n_ft_rows, PSQT_BKTS), dtype=np.uint32)
                for k in range(n_ft_rows):
                    fi_arr[k]    = struct.unpack('<I', f.read(4))[0]
                    ft_w_arr[k]  = np.frombuffer(f.read(HALF_DIMS * 4), dtype=np.float32) / TDLEAF_SCALE
                    ft_c_arr[k]  = np.frombuffer(f.read(HALF_DIMS * 4), dtype=np.uint32)
                    ps_w_arr[k]  = np.frombuffer(f.read(PSQT_BKTS * 4), dtype=np.float32) / TDLEAF_SCALE
                    ps_c_arr[k]  = np.frombuffer(f.read(PSQT_BKTS * 4), dtype=np.uint32)
                data['ft_fi']     = fi_arr
                data['ft_w']      = ft_w_arr
                data['ft_w_cnt']  = ft_c_arr
                data['psqt_w']    = ps_w_arr
                data['psqt_cnt']  = ps_c_arr
            else:
                data['ft_fi'] = np.empty(0, dtype=np.uint32)
            data['n_ft_rows'] = n_ft_rows

            # v4: FT bias section (appended after sparse FT/PSQT rows)
            if version == TDLEAF_VERSION4:
                ft_b_raw = np.frombuffer(f.read(HALF_DIMS * 4), dtype=np.float32).copy()
                ft_b_cnt = np.frombuffer(f.read(HALF_DIMS * 4), dtype=np.uint32).copy()
                if ft_b_raw.shape == (HALF_DIMS,) and ft_b_cnt.shape == (HALF_DIMS,):
                    data['ft_bias_learned']     = ft_b_raw / TDLEAF_SCALE
                    data['ft_bias_learned_cnt'] = ft_b_cnt

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

def _wstats(arr, is_int32=False):
    """Return (n, min, max, mean, std, pct_zero) for a weight array."""
    a = arr.astype(np.int64 if is_int32 else np.int32)
    return dict(n=a.size,
                wmin=int(a.min()), wmax=int(a.max()),
                mean=float(a.mean()), std=float(a.std()),
                pct_zero=100.0 * float(np.mean(a == 0)))


def print_summary(orig, upd, ft_data=None):
    has_counts = upd.get('_has_counts', False)

    # -----------------------------------------------------------------------
    # FC weight delta table
    # -----------------------------------------------------------------------
    print("\n━━━━  FC layer changes (trained + persisted to .tdleaf.bin)  ━━━━")
    print("\n┌──────────┬────────────────┬───────────────┬───────────────┬─────────────────┐")
    print("│  Layer   │    Changed     │   % Changed   │    Δ range    │   mean ± std    │")
    print("├──────────┼────────────────┼───────────────┼───────────────┼─────────────────┤")

    for layer_name, key_w, key_b in [
            ('FC0 wts',  'fc0_w',   'fc0_bias'),
            ('FC1 wts',  'fc1_w',   'fc1_bias'),
            ('FC2 wts',  'fc2_w',   'fc2_bias'),
    ]:
        all_o = np.concatenate(orig[key_w])
        all_u = np.concatenate(upd[key_w])
        s = delta_stats(all_o, all_u)
        print(f"│ {layer_name:<8} │ {s['n_changed']:>6}/{s['n']:<7} │ {s['pct']:>12.2f}% │"
              f" [{s['dmin']:+4d},{s['dmax']:+4d}]   │ {s['dmean']:+7.3f} ± {s['dstd']:.3f} │")

        # biases (int32)
        all_ob = np.concatenate(orig[key_b])
        all_ub = np.concatenate(upd[key_b])
        sb = bias_delta_stats(all_ob, all_ub)
        print(f"│ {layer_name[:-3]+'bis':<8} │ {sb['n_changed']:>6}/{sb['n']:<7} │ {sb['pct']:>12.2f}% │"
              f" [{sb['dmin']:+4d},{sb['dmax']:+4d}]   │ {sb['dmean']:+7.3f} ± {sb['dstd']:.3f} │")

    print("└──────────┴────────────────┴───────────────┴───────────────┴─────────────────┘")

    # Per-stack breakdown
    print("\nPer-stack FC1 weight changes:")
    print(f"  {'Stack':<7} {'Changed':>10} {'%':>8} {'Δ min':>8} {'Δ max':>8} {'|Δ| mean':>10}")
    for s in range(N_STACKS):
        st = delta_stats(orig['fc1_w'][s], upd['fc1_w'][s])
        print(f"  Stack {s}  {st['n_changed']:>6}/{st['n']:<4} {st['pct']:>7.2f}%"
              f"  {st['dmin']:>6}  {st['dmax']:>6}  {abs(st['dmean']):>9.3f}")

    # Update count summary (v2/v3)
    if has_counts:
        print("\nFC update count summary (times each weight was updated across sessions):")
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

    # -----------------------------------------------------------------------
    # FT / PSQT summary (trained + persisted as of v3)
    # -----------------------------------------------------------------------
    print()
    n_ft_rows = upd.get('n_ft_rows', None)
    if n_ft_rows is not None:
        print(f"━━━━  FT / PSQT weights (trained + persisted to .tdleaf.bin v3)  ━━━━")
        print(f"      {n_ft_rows:,} of {FT_INPUTS:,} feature rows have training history")
    else:
        print("━━━━  FT / PSQT weights (v3 data not in .tdleaf.bin)  ━━━━")
        print("      Run more training games — FT/PSQT saved starting with v3 format")
    print()

    if ft_data is None:
        print("  (baseline .nnue stats not read)")
    else:
        # FT biases baseline
        fb  = ft_data['ft_bias']
        fbs = _wstats(fb)
        print(f"  FT biases (baseline)  {fbs['n']:>7,} int16   "
              f"range [{fbs['wmin']:+6d}, {fbs['wmax']:+6d}]  "
              f"mean {fbs['mean']:+7.1f} ± {fbs['std']:.1f}")

        # FT weights: if v3 data is present, show learned values; else baseline
        if n_ft_rows and n_ft_rows > 0 and 'ft_w' in upd:
            fw_learned = upd['ft_w']          # [n_ft_rows, HALF_DIMS]
            fw_cnt     = upd['ft_w_cnt']      # [n_ft_rows, HALF_DIMS]
            fw_flat    = fw_learned.ravel()
            cnt_flat   = fw_cnt.ravel()
            n_updated  = int(np.sum(cnt_flat > 0))
            max_cnt    = int(cnt_flat.max()) if len(cnt_flat) else 0
            nz_cnt     = cnt_flat[cnt_flat > 0]
            print(f"  FT weights (learned)  {n_ft_rows:>7,} rows × {HALF_DIMS} dims = "
                  f"{n_ft_rows * HALF_DIMS:,} values")
            print(f"    weight range  [{fw_flat.min():+.3f}, {fw_flat.max():+.3f}]  "
                  f"mean {fw_flat.mean():+.3f} ± {fw_flat.std():.3f}")
            print(f"    update counts  ever-updated: {n_updated:,}/{n_ft_rows*HALF_DIMS:,}  "
                  f"max: {max_cnt}  mean(>0): {nz_cnt.mean():.1f}")
        else:
            ft_total = FT_INPUTS * HALF_DIMS
            if ft_data.get('ft_w_read'):
                fw  = ft_data['ft_w']
                fws = _wstats(fw)
                print(f"  FT weights (baseline) {fws['n']:>12,} int16   "
                      f"range [{fws['wmin']:+6d}, {fws['wmax']:+6d}]  "
                      f"mean {fws['mean']:+.3f} ± {fws['std']:.3f}")
            else:
                print(f"  FT weights (baseline) {ft_total:>12,} int16   "
                      f"(not read — use --ft-weights to load)")

        # PSQT: if v3, compare learned vs baseline for dirty rows
        pw_base = ft_data['psqt_w']   # [FT_INPUTS, PSQT_BKTS], baseline int32 → float
        pws = _wstats(pw_base, is_int32=True)
        print(f"  PSQT baseline         {pws['n']:>7,} int32   "
              f"range [{pws['wmin']:+6d}, {pws['wmax']:+6d}]  "
              f"mean {pws['mean']:+7.1f} ± {pws['std']:.1f}")

        if n_ft_rows and n_ft_rows > 0 and 'psqt_w' in upd:
            ps_learned = upd['psqt_w']    # [n_ft_rows, PSQT_BKTS]  float at int32 scale
            ps_cnt     = upd['psqt_cnt']  # [n_ft_rows, PSQT_BKTS]
            fi_arr     = upd['ft_fi']     # [n_ft_rows]
            # Compute delta vs baseline for the dirty rows
            baseline_dirty = pw_base[fi_arr]  # [n_ft_rows, PSQT_BKTS] — float32 from int32
            delta = ps_learned - baseline_dirty.astype(np.float32)
            cnt_flat = ps_cnt.ravel()
            nz_cnt   = cnt_flat[cnt_flat > 0]
            print(f"  PSQT learned          {n_ft_rows:>7,} rows × {PSQT_BKTS} buckets")
            print(f"    Δ range  [{delta.min():+.1f}, {delta.max():+.1f}]  "
                  f"mean Δ {delta.mean():+.3f} ± {delta.std():.3f}")
            print(f"    update counts  max: {int(cnt_flat.max())}  "
                  f"mean(>0): {nz_cnt.mean():.1f}" if len(nz_cnt) else
                  f"    update counts  (none)")

            # Per-bucket delta stats for learned rows
            print()
            print(f"  PSQT Δ per-bucket (learned − baseline, dirty rows only):")
            print(f"  {'Bucket':>8} {'Rows':>8} {'Δ min':>10} {'Δ max':>10} "
                  f"{'mean Δ':>10} {'max cnt':>9} {'mean cnt(>0)':>14}")
            for b in range(PSQT_BKTS):
                d_col = delta[:, b]
                c_col = ps_cnt[:, b]
                nz_c  = c_col[c_col > 0]
                print(f"  {b:>8} {n_ft_rows:>8,} {d_col.min():>10.1f} {d_col.max():>10.1f}"
                      f" {d_col.mean():>10.3f} {int(c_col.max()):>9}"
                      f" {nz_c.mean():>14.1f}" if len(nz_c) else
                      f"  {b:>8} {n_ft_rows:>8,} {d_col.min():>10.1f} {d_col.max():>10.1f}"
                      f" {d_col.mean():>10.3f} {int(c_col.max()):>9}         —")
        else:
            # No v3 learned data — show baseline per-bucket
            print()
            print(f"  PSQT per-bucket baseline (bucket 0 = most material):")
            print(f"  {'Bucket':>8} {'Count':>10} {'Min':>8} {'Max':>8} "
                  f"{'Mean':>10} {'Std':>10} {'% zero':>8}")
            for b in range(PSQT_BKTS):
                col = pw_base[:, b]
                nz  = int(np.sum(col == 0))
                print(f"  {b:>8} {len(col):>10,} {int(col.min()):>8} {int(col.max()):>8}"
                      f" {col.mean():>10.1f} {col.std():>10.1f} {100*nz/len(col):>7.1f}%")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overview(orig, upd, save):
    """
    Col 0: two stacked panels per FC layer — top=file1 (blue), bottom=file2 (red).
            Nested GridSpec keeps the pair tightly spaced with no gap between them.
    Col 1: delta distribution  |  Col 2: per-stack bar  (one panel each, full row height).
    Outer GridSpec(3×3) controls spacing between FC-layer groups.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    nnue_name   = orig.get('_name', 'original .nnue')
    tdleaf_name = upd.get('_name',  'updated .tdleaf.bin')

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f'FC weight comparison\n'
                 f'blue = {nnue_name}   red = {tdleaf_name}   orange = delta',
                 fontsize=11)

    # Outer grid: 3 FC-layer rows × 3 cols.
    # hspace controls vertical gap *between* FC-layer groups.
    gs = GridSpec(3, 3, figure=fig,
                  hspace=0.55, wspace=0.35,
                  top=0.88, bottom=0.06, left=0.07, right=0.97)

    layers = [
        ('FC0 weights (16×1024)',  'fc0_w'),
        ('FC1 weights (32×32)',    'fc1_w'),
        ('FC2 weights (32)',       'fc2_w'),
    ]

    bins      = np.arange(-128, 130) - 0.5
    last_row  = len(layers) - 1

    for row, (layer_name, key) in enumerate(layers):
        all_orig = np.concatenate(orig[key]).ravel().astype(np.int32)
        all_upd  = np.concatenate(upd[key]).ravel().astype(np.int32)
        delta    = all_upd - all_orig

        # --- col 0: nested 2-row subgrid, near-zero inner gap ---
        gs_inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[row, 0], hspace=0.08)

        ax_top = fig.add_subplot(gs_inner[0])
        ax_top.hist(all_orig, bins=bins, color='steelblue', alpha=0.75, density=True)
        ax_top.set_title(layer_name, fontsize=9, fontweight='bold')
        ax_top.set_ylabel('density', fontsize=8)
        ax_top.tick_params(labelbottom=False, labelsize=7)
        ax_top.set_xlim(-130, 130)
        ax_top.text(0.98, 0.90, nnue_name, transform=ax_top.transAxes,
                    fontsize=7, ha='right', va='top', color='steelblue')

        ax_bot = fig.add_subplot(gs_inner[1], sharex=ax_top)
        ax_bot.hist(all_upd, bins=bins, color='lightcoral', alpha=0.75, density=True)
        ax_bot.set_ylabel('density', fontsize=8)
        ax_bot.tick_params(labelsize=7)
        ax_bot.set_xlim(-130, 130)
        ax_bot.text(0.98, 0.90, tdleaf_name, transform=ax_bot.transAxes,
                    fontsize=7, ha='right', va='top', color='firebrick')
        # Only show x-axis label on bottom row to avoid inter-group clutter
        if row == last_row:
            ax_bot.set_xlabel('weight value', fontsize=8)

        # --- col 1: delta distribution ---
        ax1 = fig.add_subplot(gs[row, 1])
        d_max  = max(abs(int(delta.min())), abs(int(delta.max())), 1)
        d_bins = np.arange(-d_max - 1, d_max + 2) - 0.5
        ax1.hist(delta, bins=d_bins, color='coral', alpha=0.85)
        n_nz = int(np.sum(delta != 0))
        ax1.set_title(f'{layer_name}\nΔ  ({n_nz}/{delta.size} changed)', fontsize=9)
        ax1.set_ylabel('count', fontsize=8)
        ax1.axvline(0, color='k', linewidth=0.9, linestyle='--')
        ax1.tick_params(labelsize=7)
        if row == last_row:
            ax1.set_xlabel('Δ value (updated − original)', fontsize=8)

        # --- col 2: per-stack changed fraction + max|Δ| ---
        ax2 = fig.add_subplot(gs[row, 2])
        pcts, maxds = [], []
        for s in range(N_STACKS):
            d = upd[key][s].astype(np.int32) - orig[key][s].astype(np.int32)
            pcts.append(100 * np.mean(d != 0))
            maxds.append(int(np.max(np.abs(d))))
        x = np.arange(N_STACKS)
        ax2.bar(x, pcts, color='mediumseagreen', alpha=0.80)
        ax2r = ax2.twinx()
        ax2r.plot(x, maxds, 'r^--', markersize=7, label='max |Δ|')
        ax2r.set_ylabel('max |Δ|', color='red', fontsize=9)
        ax2r.tick_params(axis='y', colors='red', labelsize=7)
        ax2r.legend(loc='upper right', fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'S{i}' for i in x], fontsize=7)
        ax2.set_title(f'{layer_name}\n% changed per stack', fontsize=9)
        ax2.set_ylabel('% weights changed', fontsize=8)
        ax2.tick_params(labelsize=7)
        if row == last_row:
            ax2.set_xlabel('stack', fontsize=8)

    if save:
        fig.savefig('fc_compare_overview.png', dpi=150)
        print("Saved fc_compare_overview.png")
    return fig


def plot_ft_overview(orig, upd, ft_data, save):
    """
    Page 2: FT bias and FT weight distributions.
    Row 0: FT biases (baseline only — not trained by TDLeaf); col 1 = trained feature index
           coverage; col 2 = per-feature max update count.
    Row 1: FT weights — col 0 split (baseline/learned), col 1 delta or learned dist,
           col 2 update count distribution.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    nnue_name   = orig.get('_name', 'original .nnue')
    tdleaf_name = upd.get('_name',  'updated .tdleaf.bin')

    n_ft_rows       = upd.get('n_ft_rows', None)
    has_v3          = n_ft_rows is not None and n_ft_rows > 0
    has_ft_baseline = ft_data.get('ft_w_read', False)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f'FT layer comparison\n'
                 f'blue = {nnue_name}   red = {tdleaf_name}',
                 fontsize=11)

    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.55, wspace=0.35,
                  top=0.88, bottom=0.06, left=0.07, right=0.97)

    # -----------------------------------------------------------------------
    # Row 0: FT biases (baseline only)
    # -----------------------------------------------------------------------
    gs_inner0 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], hspace=0.08)

    ax_top = fig.add_subplot(gs_inner0[0])
    fb      = ft_data['ft_bias'].astype(np.int32)
    bmax    = max(abs(int(fb.min())), abs(int(fb.max())), 1)
    # Widen range to cover learned values so neither panel clips outliers
    if 'ft_bias_learned' in upd:
        fb_learned = upd['ft_bias_learned'].astype(np.float32)
        bmax = max(bmax, abs(float(fb_learned.min())), abs(float(fb_learned.max())))
    b_bins  = np.linspace(-bmax * 1.05, bmax * 1.05, 80)
    ax_top.hist(fb, bins=b_bins, color='steelblue', alpha=0.75, density=True)
    ax_top.set_title('FT biases (1024 int16)', fontsize=9, fontweight='bold')
    ax_top.set_ylabel('density', fontsize=8)
    ax_top.tick_params(labelbottom=False, labelsize=7)
    ax_top.text(0.98, 0.90, nnue_name, transform=ax_top.transAxes,
                fontsize=7, ha='right', va='top', color='steelblue')

    ax_bot = fig.add_subplot(gs_inner0[1], sharex=ax_top)
    if 'ft_bias_learned' in upd:
        ax_bot.hist(fb_learned, bins=b_bins, color='lightcoral', alpha=0.75, density=True)
        ax_bot.text(0.98, 0.90, tdleaf_name, transform=ax_bot.transAxes,
                    fontsize=7, ha='right', va='top', color='firebrick')
    else:
        ax_bot.text(0.5, 0.5, 'FT biases not trained\n(v3 or older .tdleaf.bin)',
                    transform=ax_bot.transAxes, fontsize=8,
                    ha='center', va='center', color='gray', style='italic')
    ax_bot.set_ylabel('density', fontsize=8)
    ax_bot.tick_params(labelsize=7)

    # Col 1: FT bias delta distribution (v4) or feature index coverage (v3/fallback)
    ax1 = fig.add_subplot(gs[0, 1])
    if 'ft_bias_learned' in upd:
        fb_learned = upd['ft_bias_learned']
        delta_b    = fb_learned - ft_data['ft_bias'].astype(np.float32)
        dmax  = max(abs(float(delta_b.min())), abs(float(delta_b.max())), 0.1)
        d_bins = np.linspace(-dmax * 1.05, dmax * 1.05, 80)
        ax1.hist(delta_b, bins=d_bins, color='coral', alpha=0.85)
        n_nz = int(np.sum(np.abs(delta_b) > 0.5))
        ax1.set_title(f'FT biases\nΔ learned − baseline  ({n_nz}/{HALF_DIMS} shifted)',
                      fontsize=9)
        ax1.set_ylabel('count', fontsize=8)
        ax1.axvline(0, color='k', linewidth=0.9, linestyle='--')
        ax1.set_xlabel('Δ value', fontsize=8)
    elif has_v3:
        fi_arr = upd['ft_fi']
        ax1.hist(fi_arr, bins=60, color='steelblue', alpha=0.80)
        ax1.set_title(f'FT weights\nTrained feature indices ({n_ft_rows:,}/{FT_INPUTS:,})',
                      fontsize=9)
        ax1.set_ylabel('count', fontsize=8)
        ax1.set_xlabel('feature index', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No v3/v4 training data\nin .tdleaf.bin',
                 transform=ax1.transAxes, fontsize=9, ha='center', va='center', color='gray')
        ax1.set_title('FT biases\nDelta distribution', fontsize=9)
    ax1.tick_params(labelsize=7)

    # Col 2: FT bias update count (v4) or per-feature max update count (v3/fallback)
    ax2 = fig.add_subplot(gs[0, 2])
    if 'ft_bias_learned_cnt' in upd:
        cnt_nz = upd['ft_bias_learned_cnt']
        cnt_nz = cnt_nz[cnt_nz > 0]
        if len(cnt_nz):
            ax2.hist(cnt_nz, bins=40, color='mediumseagreen', alpha=0.80)
            ax2.set_title('FT biases\nUpdate count distribution (>0)', fontsize=9)
            ax2.set_ylabel('count', fontsize=8)
            ax2.set_xlabel('update count', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No bias updates yet', transform=ax2.transAxes,
                     fontsize=9, ha='center', va='center', color='gray')
            ax2.set_title('FT biases\nUpdate count distribution', fontsize=9)
    elif has_v3 and 'ft_w_cnt' in upd:
        max_cnts = upd['ft_w_cnt'].max(axis=1)
        ax2.hist(max_cnts, bins=40, color='mediumseagreen', alpha=0.80)
        ax2.set_title('FT weights\nMax update count per feature row', fontsize=9)
        ax2.set_ylabel('count', fontsize=8)
        ax2.set_xlabel('max updates', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No v3/v4 training data', transform=ax2.transAxes,
                 fontsize=9, ha='center', va='center', color='gray')
        ax2.set_title('FT biases\nUpdate count distribution', fontsize=9)
    ax2.tick_params(labelsize=7)

    # -----------------------------------------------------------------------
    # Row 1: FT weights
    # -----------------------------------------------------------------------
    gs_inner1 = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.08)

    # Top panel: baseline (if --ft-weights loaded)
    ax_top1 = fig.add_subplot(gs_inner1[0])
    if has_ft_baseline:
        fw_base = ft_data['ft_w'].astype(np.int32)
        fmax    = max(abs(int(fw_base.min())), abs(int(fw_base.max())), 1)
        fw_bins = np.linspace(-fmax * 1.05, fmax * 1.05, 80)
        ax_top1.hist(fw_base.ravel(), bins=fw_bins, color='steelblue', alpha=0.75, density=True)
        ax_top1.text(0.98, 0.90, nnue_name, transform=ax_top1.transAxes,
                     fontsize=7, ha='right', va='top', color='steelblue')
    else:
        fw_bins = None
        ax_top1.text(0.5, 0.5, 'Baseline not loaded\n(use --ft-weights)',
                     transform=ax_top1.transAxes, fontsize=8,
                     ha='center', va='center', color='gray', style='italic')
    ax_top1.set_title('FT weights (22528×1024 int16)', fontsize=9, fontweight='bold')
    ax_top1.set_ylabel('density', fontsize=8)
    ax_top1.tick_params(labelbottom=False, labelsize=7)

    # Bottom panel: learned distribution for dirty rows
    ax_bot1 = fig.add_subplot(gs_inner1[1],
                               sharex=ax_top1 if has_ft_baseline else None)
    if has_v3 and 'ft_w' in upd:
        fw_learned = upd['ft_w'].ravel()
        if fw_bins is None:
            fw_max_l = max(abs(float(fw_learned.min())), abs(float(fw_learned.max())), 1.0)
            fw_bins_l = np.linspace(-fw_max_l * 1.05, fw_max_l * 1.05, 80)
        else:
            fw_bins_l = fw_bins
        ax_bot1.hist(fw_learned, bins=fw_bins_l, color='lightcoral', alpha=0.75, density=True)
        ax_bot1.text(0.98, 0.90, tdleaf_name, transform=ax_bot1.transAxes,
                     fontsize=7, ha='right', va='top', color='firebrick')
    else:
        ax_bot1.text(0.5, 0.5, 'No FT weight training data',
                     transform=ax_bot1.transAxes, fontsize=8,
                     ha='center', va='center', color='gray', style='italic')
    ax_bot1.set_ylabel('density', fontsize=8)
    ax_bot1.set_xlabel('weight value', fontsize=8)
    ax_bot1.tick_params(labelsize=7)

    # Col 1: Delta (if both available) else learned distribution
    ax3 = fig.add_subplot(gs[1, 1])
    if has_v3 and 'ft_w' in upd and has_ft_baseline:
        fi_arr   = upd['ft_fi']
        baseline = ft_data['ft_w'].reshape(FT_INPUTS, HALF_DIMS)[fi_arr].ravel().astype(np.float32)
        learned  = upd['ft_w'].ravel()
        delta    = learned - baseline
        dmax     = max(abs(float(delta.min())), abs(float(delta.max())), 0.1)
        d_bins   = np.linspace(-dmax * 1.05, dmax * 1.05, 80)
        ax3.hist(delta, bins=d_bins, color='coral', alpha=0.85)
        n_nz = int(np.sum(np.abs(delta) > 0.5))
        ax3.set_title(f'FT weights\nΔ learned − baseline  ({n_nz:,} shifted)', fontsize=9)
        ax3.set_ylabel('count', fontsize=8)
        ax3.axvline(0, color='k', linewidth=0.9, linestyle='--')
    elif has_v3 and 'ft_w' in upd:
        fw_learned = upd['ft_w'].ravel()
        fw_max_l   = max(abs(float(fw_learned.min())), abs(float(fw_learned.max())), 1.0)
        fw_bins_l  = np.linspace(-fw_max_l * 1.05, fw_max_l * 1.05, 80)
        ax3.hist(fw_learned, bins=fw_bins_l, color='coral', alpha=0.85, density=True)
        ax3.set_title(f'FT weights\nLearned distribution ({n_ft_rows:,} rows)', fontsize=9)
        ax3.set_ylabel('density', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No FT training data', transform=ax3.transAxes,
                 fontsize=9, ha='center', va='center', color='gray')
        ax3.set_title('FT weights\nDelta distribution', fontsize=9)
    ax3.set_xlabel('Δ value', fontsize=8)
    ax3.tick_params(labelsize=7)

    # Col 2: Update count distribution (non-zero counts)
    ax4 = fig.add_subplot(gs[1, 2])
    if has_v3 and 'ft_w_cnt' in upd:
        cnt_nz = upd['ft_w_cnt'].ravel()
        cnt_nz = cnt_nz[cnt_nz > 0]
        if len(cnt_nz):
            ax4.hist(cnt_nz, bins=40, color='mediumseagreen', alpha=0.80)
            ax4.set_title('FT weights\nUpdate count distribution (>0)', fontsize=9)
            ax4.set_ylabel('count', fontsize=8)
            ax4.set_xlabel('update count', fontsize=8)
        else:
            ax4.text(0.5, 0.5, 'No weight updates recorded',
                     transform=ax4.transAxes, fontsize=9, ha='center', va='center', color='gray')
            ax4.set_title('FT weights\nUpdate count distribution', fontsize=9)
    else:
        ax4.text(0.5, 0.5, 'No v3 training data', transform=ax4.transAxes,
                 fontsize=9, ha='center', va='center', color='gray')
        ax4.set_title('FT weights\nUpdate count distribution', fontsize=9)
    ax4.tick_params(labelsize=7)

    if save:
        fig.savefig('ft_compare_overview.png', dpi=150)
        print("Saved ft_compare_overview.png")
    return fig


def plot_fc_bias_overview(orig, upd, save):
    """
    Page N: FC bias distributions and deltas (int32).
    3 rows (FC0, FC1, FC2) × 3 cols:
      Col 0: stacked panels — top=baseline (blue), bottom=learned (red),
             x-axis set by the wider of the two distributions so outliers
             in either panel are never clipped.
      Col 1: delta distribution histogram (learned − baseline).
      Col 2: per-stack scatter of individual Δ values; with so few biases
             per stack every point is visible so no outlier can hide in
             an aggregate statistic.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    nnue_name   = orig.get('_name', 'original .nnue')
    tdleaf_name = upd.get('_name',  'updated .tdleaf.bin')
    has_counts  = upd.get('_has_counts', False)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f'FC bias comparison (int32)\n'
                 f'blue = {nnue_name}   red = {tdleaf_name}   orange = delta',
                 fontsize=11)

    gs = GridSpec(3, 3, figure=fig,
                  hspace=0.55, wspace=0.35,
                  top=0.88, bottom=0.06, left=0.07, right=0.97)

    layers = [
        ('FC0 biases (16/stack)', 'fc0_bias', 'fc0_bias_cnt', L0_SIZE),
        ('FC1 biases (32/stack)', 'fc1_bias', 'fc1_bias_cnt', L1_SIZE),
        ('FC2 bias  (1/stack)',   'fc2_bias', 'fc2_bias_cnt', 1),
    ]
    last_row = len(layers) - 1

    rng = np.random.default_rng(0)   # deterministic jitter

    for row, (layer_name, key, cnt_key, n_per_stack) in enumerate(layers):
        all_orig = np.concatenate(orig[key]).astype(np.int64)
        all_upd  = np.concatenate(upd[key]).astype(np.int64)
        delta    = all_upd - all_orig

        # --- col 0: stacked histograms, x-axis = max range of both ---
        bmax   = max(abs(int(all_orig.min())), abs(int(all_orig.max())),
                     abs(int(all_upd.min())),  abs(int(all_upd.max())), 1)
        b_bins = np.linspace(-bmax * 1.05, bmax * 1.05, 60)

        gs_inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[row, 0], hspace=0.08)

        ax_top = fig.add_subplot(gs_inner[0])
        ax_top.hist(all_orig, bins=b_bins, color='steelblue', alpha=0.75, density=True)
        ax_top.set_title(layer_name, fontsize=9, fontweight='bold')
        ax_top.set_ylabel('density', fontsize=8)
        ax_top.tick_params(labelbottom=False, labelsize=7)
        ax_top.text(0.98, 0.90, nnue_name, transform=ax_top.transAxes,
                    fontsize=7, ha='right', va='top', color='steelblue')

        ax_bot = fig.add_subplot(gs_inner[1], sharex=ax_top)
        ax_bot.hist(all_upd, bins=b_bins, color='lightcoral', alpha=0.75, density=True)
        ax_bot.set_ylabel('density', fontsize=8)
        ax_bot.tick_params(labelsize=7)
        ax_bot.text(0.98, 0.90, tdleaf_name, transform=ax_bot.transAxes,
                    fontsize=7, ha='right', va='top', color='firebrick')
        if row == last_row:
            ax_bot.set_xlabel('bias value (int32)', fontsize=8)

        # --- col 1: delta distribution ---
        ax1 = fig.add_subplot(gs[row, 1])
        d_max  = max(abs(int(delta.min())), abs(int(delta.max())), 1)
        d_bins = np.linspace(-d_max * 1.05, d_max * 1.05, min(60, 2 * d_max + 3))
        n_nz   = int(np.sum(delta != 0))
        ax1.hist(delta, bins=d_bins, color='coral', alpha=0.85)
        ax1.set_title(f'{layer_name}\nΔ  ({n_nz}/{delta.size} changed)', fontsize=9)
        ax1.set_ylabel('count', fontsize=8)
        ax1.axvline(0, color='k', linewidth=0.9, linestyle='--')
        ax1.tick_params(labelsize=7)
        if row == last_row:
            ax1.set_xlabel('Δ value (int32, learned − baseline)', fontsize=8)

        # --- col 2: per-stack scatter of individual Δ values ---
        ax2 = fig.add_subplot(gs[row, 2])
        x_vals, y_vals = [], []
        for s in range(N_STACKS):
            d_s = upd[key][s].astype(np.int64) - orig[key][s].astype(np.int64)
            n   = len(d_s)
            # horizontal jitter so overlapping points are visible
            jitter = rng.uniform(-0.25, 0.25, n) if n > 1 else np.zeros(1)
            x_vals.append(np.full(n, s, dtype=float) + jitter)
            y_vals.append(d_s)
        ax2.scatter(np.concatenate(x_vals), np.concatenate(y_vals),
                    s=18, alpha=0.70, color='coral', linewidths=0)
        ax2.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax2.set_xticks(np.arange(N_STACKS))
        ax2.set_xticklabels([f'S{i}' for i in range(N_STACKS)], fontsize=7)
        ax2.set_title(f'{layer_name}\nΔ per stack (each point = one bias)', fontsize=9)
        ax2.set_ylabel('Δ (int32)', fontsize=8)
        ax2.tick_params(labelsize=7)
        # Annotate with update counts if available
        if has_counts and cnt_key in upd:
            cnts = np.concatenate(upd[cnt_key])
            ever = int(np.sum(cnts > 0))
            mx   = int(cnts.max())
            ax2.text(0.02, 0.97,
                     f'updated: {ever}/{cnts.size}  max cnt: {mx}',
                     transform=ax2.transAxes, fontsize=7,
                     va='top', ha='left', color='dimgray')
        if row == last_row:
            ax2.set_xlabel('stack', fontsize=8)

    if save:
        fig.savefig('fc_bias_compare_overview.png', dpi=150)
        print("Saved fc_bias_compare_overview.png")
    return fig


def plot_psqt_overview(orig, upd, ft_data, save):
    """
    Page 3: PSQT weight distributions.
    Single row: col 0 split (baseline/learned), col 1 delta distribution,
    col 2 per-bucket mean delta bar chart.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    nnue_name   = orig.get('_name', 'original .nnue')
    tdleaf_name = upd.get('_name',  'updated .tdleaf.bin')

    n_ft_rows = upd.get('n_ft_rows', None)
    has_v3    = n_ft_rows is not None and n_ft_rows > 0

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f'PSQT weight comparison\n'
                 f'blue = {nnue_name}   red = {tdleaf_name}   orange = delta',
                 fontsize=11)

    gs = GridSpec(1, 3, figure=fig,
                  wspace=0.35,
                  top=0.82, bottom=0.14, left=0.07, right=0.97)

    pw_base = ft_data['psqt_w']   # [FT_INPUTS, PSQT_BKTS] int32

    # --- col 0: split — baseline (top) / learned (bottom) ---
    gs_inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], hspace=0.08)

    ax_top = fig.add_subplot(gs_inner[0])
    pb_flat = pw_base.ravel().astype(np.int64)
    p_max   = max(abs(int(pb_flat.min())), abs(int(pb_flat.max())), 1)
    p_bins  = np.linspace(-p_max * 1.05, p_max * 1.05, 80)
    ax_top.hist(pb_flat, bins=p_bins, color='steelblue', alpha=0.75, density=True)
    ax_top.set_title('PSQT weights (22528×8 int32)', fontsize=9, fontweight='bold')
    ax_top.set_ylabel('density', fontsize=8)
    ax_top.tick_params(labelbottom=False, labelsize=7)
    ax_top.text(0.98, 0.90, nnue_name, transform=ax_top.transAxes,
                fontsize=7, ha='right', va='top', color='steelblue')

    ax_bot = fig.add_subplot(gs_inner[1], sharex=ax_top)
    if has_v3 and 'psqt_w' in upd:
        ps_flat = upd['psqt_w'].ravel()
        ax_bot.hist(ps_flat, bins=p_bins, color='lightcoral', alpha=0.75, density=True)
        ax_bot.text(0.98, 0.90, tdleaf_name, transform=ax_bot.transAxes,
                    fontsize=7, ha='right', va='top', color='firebrick')
    else:
        ax_bot.text(0.5, 0.5, 'No PSQT training data', transform=ax_bot.transAxes,
                    fontsize=8, ha='center', va='center', color='gray', style='italic')
    ax_bot.set_ylabel('density', fontsize=8)
    ax_bot.set_xlabel('weight value (int32 units)', fontsize=8)
    ax_bot.tick_params(labelsize=7)

    # --- col 1: delta distribution ---
    ax1 = fig.add_subplot(gs[0, 1])
    if has_v3 and 'psqt_w' in upd:
        fi_arr     = upd['ft_fi']
        baseline_d = pw_base[fi_arr].astype(np.float32).ravel()
        learned_d  = upd['psqt_w'].ravel()
        delta      = learned_d - baseline_d
        dmax   = max(abs(float(delta.min())), abs(float(delta.max())), 0.1)
        d_bins = np.linspace(-dmax * 1.05, dmax * 1.05, 80)
        ax1.hist(delta, bins=d_bins, color='coral', alpha=0.85)
        n_nz = int(np.sum(np.abs(delta) > 0.5))
        ax1.set_title(f'PSQT weights\nΔ learned − baseline  ({n_nz:,} shifted)', fontsize=9)
        ax1.set_ylabel('count', fontsize=8)
        ax1.axvline(0, color='k', linewidth=0.9, linestyle='--')
    else:
        ax1.text(0.5, 0.5, 'No PSQT training data', transform=ax1.transAxes,
                 fontsize=9, ha='center', va='center', color='gray')
        ax1.set_title('PSQT weights\nDelta distribution', fontsize=9)
    ax1.set_xlabel('Δ value (int32 units)', fontsize=8)
    ax1.tick_params(labelsize=7)

    # --- col 2: per-bucket mean delta bar chart ---
    ax2 = fig.add_subplot(gs[0, 2])
    x = np.arange(PSQT_BKTS)
    if has_v3 and 'psqt_w' in upd:
        fi_arr     = upd['ft_fi']
        baseline_b = pw_base[fi_arr].astype(np.float32)    # [n_ft_rows, PSQT_BKTS]
        delta_b    = upd['psqt_w'] - baseline_b             # [n_ft_rows, PSQT_BKTS]
        mean_delta = delta_b.mean(axis=0)
        std_delta  = delta_b.std(axis=0)
        ax2.bar(x, mean_delta, yerr=std_delta, color='mediumseagreen', alpha=0.80,
                capsize=4, error_kw={'linewidth': 1})
        ax2.axhline(0, color='k', linewidth=0.7, linestyle='--')
        ax2.set_title(f'PSQT weights\nMean Δ per bucket (±1σ)', fontsize=9)
        ax2.set_ylabel('mean Δ (int32 units)', fontsize=8)
    else:
        means = pw_base.mean(axis=0)
        stds  = pw_base.std(axis=0)
        ax2.bar(x, means, yerr=stds, color='steelblue', alpha=0.80, capsize=4,
                error_kw={'linewidth': 1})
        ax2.axhline(0, color='k', linewidth=0.7, linestyle='--')
        ax2.set_title('PSQT weights\nBaseline mean per bucket (±1σ)', fontsize=9)
        ax2.set_ylabel('mean weight (int32 units)', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'B{i}' for i in x], fontsize=7)
    ax2.set_xlabel('bucket', fontsize=8)
    ax2.tick_params(labelsize=7)

    if save:
        fig.savefig('psqt_compare_overview.png', dpi=150)
        print("Saved psqt_compare_overview.png")
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


def _plot_delta_heatmaps_unused(orig, upd, save):
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


def _plot_bias_changes_unused(orig, upd, save):
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
        description='Compare FC weights (.nnue vs .tdleaf.bin); summarise FT/PSQT trainable layers.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('nnue',   help='.nnue file (original weights)')
    ap.add_argument('tdleaf', help='.tdleaf.bin file (updated FC weights)')
    ap.add_argument('--save', action='store_true',
                    help='Save plots to PNG files instead of (or in addition to) showing')
    ap.add_argument('--no-show', action='store_true',
                    help='Do not open interactive windows (implies --save)')
    ap.add_argument('--ft-weights', action='store_true',
                    help='Also decode FT weights from .nnue (23M int16, ~30s). '
                         'Always shows FT biases and PSQT regardless.')
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

    print(f"Reading FC layers from {args.nnue} ...")
    orig = read_nnue_fc(args.nnue)
    orig['_name'] = os.path.basename(args.nnue)

    print(f"Reading FC layers from {args.tdleaf} ...")
    upd = read_tdleaf_fc(args.tdleaf)
    upd['_name'] = os.path.basename(args.tdleaf)

    print(f"Reading FT/PSQT from {args.nnue} ...")
    ft_data = read_nnue_ft(args.nnue, read_ft_weights=args.ft_weights)

    print_summary(orig, upd, ft_data)

    plot_overview(orig, upd, args.save)
    plot_fc_bias_overview(orig, upd, args.save)
    plot_ft_overview(orig, upd, ft_data, args.save)
    plot_psqt_overview(orig, upd, ft_data, args.save)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
