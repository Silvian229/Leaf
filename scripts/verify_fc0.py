#!/usr/bin/env python3
"""
Verify EXchess FC0 computation against the raw .nnue file.

Usage:
  1. Run EXchess with NNUE_DEBUG_VERBOSE=1 and capture the l0_in output.
  2. Set L0_IN below from that output.
  3. python3 verify_fc0.py

The script parses nn-ad9b42354671.nnue, skips the LEB128-compressed FT
section, navigates to stack 7's FC0 section, and computes fc0_raw from
the given l0_in to compare with EXchess's output.
"""

import struct, sys, os

NET_PATH = "nn-ad9b42354671.nnue"
HALF_DIMS   = 1024
FT_INPUTS   = 22528
PSQT_BKTS   = 8
LAYER_STACKS = 8
L0_SIZE     = 16
L0_INPUT    = 1024
L1_PADDED   = 32
L2_PADDED   = 32

# ------------------------------------------------------------------
# Paste l0_in from NNUE_DEBUG_VERBOSE output here.
# Format: sparse dict {index: value}, all others are 0.
# (Run: printf 'score\nquit\n' | NNUE_DEBUG_VERBOSE=1 ./EXchess_vXXX)
# ------------------------------------------------------------------
L0_IN_SPARSE = {}   # will be filled from command-line or hardcoded below

# Hardcoded from last debug run (starting position):
HARDCODED_L0_IN = {
    # filled after running the build — leave empty to read from stdin
}

# fc0_raw from EXchess (starting position, stack 7):
EXCHESS_FC0_RAW = [6158, -3377, 2464, 2299, -7790, -2787, 219, 7333,
                   4937, 14302, -2314, 4713, -2947, -5710, -1691, 3564]

# ------------------------------------------------------------------
# LEB128 decompressor
# ------------------------------------------------------------------
def read_leb128_block(f, dtype_size, count):
    magic = f.read(17)
    if magic != b"COMPRESSED_LEB128":
        f.seek(-17, 1)
        raw = f.read(dtype_size * count)
        fmt = {1: 'b', 2: 'h', 4: 'i'}[dtype_size]
        return list(struct.unpack(f'<{count}{fmt}', raw))

    nbytes = struct.unpack('<I', f.read(4))[0]
    cbuf   = f.read(nbytes)
    pos    = 0
    vals   = []
    signed = True
    bits   = dtype_size * 8
    for _ in range(count):
        val   = 0
        shift = 0
        while True:
            b = cbuf[pos]; pos += 1
            val |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        if signed and (cbuf[pos-1] & 0x40):
            val |= -(1 << shift)
        if bits == 16:
            val = val & 0xFFFF
            if val >= 0x8000: val -= 0x10000
        elif bits == 32:
            val = val & 0xFFFFFFFF
            if val >= 0x80000000: val -= 0x100000000
        vals.append(val)
    return vals

# ------------------------------------------------------------------
# Parse net file up to stacks
# ------------------------------------------------------------------
def parse_net(path):
    with open(path, 'rb') as f:
        # Header
        version, file_hash, desc_size = struct.unpack('<III', f.read(12))
        desc = f.read(desc_size)
        print(f"Net: version=0x{version:08X}  desc={desc.decode(errors='replace')}")

        # FT section
        ft_hash = struct.unpack('<I', f.read(4))[0]
        print(f"FT hash: 0x{ft_hash:08X}")

        print("Reading FT biases ...")
        ft_biases = read_leb128_block(f, 2, HALF_DIMS)
        print(f"  bias[0..3] = {ft_biases[:4]}")

        print("Reading FT weights ...")
        ft_weights = read_leb128_block(f, 2, FT_INPUTS * HALF_DIMS)
        print(f"  weight[0..3] = {ft_weights[:4]}")

        print("Reading PSQT weights ...")
        psqt_weights = read_leb128_block(f, 4, FT_INPUTS * PSQT_BKTS)
        print(f"  psqt[0..3] = {psqt_weights[:4]}")

        # Layer stacks — raw binary (no LEB128)
        stacks = []
        for s in range(LAYER_STACKS):
            stack_hash = struct.unpack('<I', f.read(4))[0]

            # FC0 biases: L0_SIZE * int32
            fc0_bias = list(struct.unpack(f'<{L0_SIZE}i', f.read(L0_SIZE * 4)))

            # FC0 weights: L0_SIZE * L0_INPUT int8 (output-major)
            fc0_wt_raw = f.read(L0_SIZE * L0_INPUT)
            fc0_wt = [struct.unpack('b', bytes([fc0_wt_raw[i]]))[0]
                      for i in range(L0_SIZE * L0_INPUT)]

            # FC1 biases: L1_PADDED * int32 (only 32 used)
            fc1_bias = list(struct.unpack(f'<{L1_PADDED}i', f.read(L1_PADDED * 4)))

            # FC1 weights: L1_PADDED * L1_PADDED int8
            fc1_wt_raw = f.read(L1_PADDED * L1_PADDED)
            fc1_wt = [struct.unpack('b', bytes([fc1_wt_raw[i]]))[0]
                      for i in range(L1_PADDED * L1_PADDED)]

            # FC2 bias: 1 * int32
            fc2_bias = struct.unpack('<i', f.read(4))[0]

            # FC2 weights: L2_PADDED int8
            fc2_wt_raw = f.read(L2_PADDED)
            fc2_wt = [struct.unpack('b', bytes([fc2_wt_raw[i]]))[0]
                      for i in range(L2_PADDED)]

            stacks.append({
                'hash':     stack_hash,
                'fc0_bias': fc0_bias,
                'fc0_wt':   fc0_wt,   # [out * L0_INPUT + in] = out-major
                'fc1_bias': fc1_bias,
                'fc1_wt':   fc1_wt,   # [out * L1_PADDED + in] = out-major
                'fc2_bias': fc2_bias,
                'fc2_wt':   fc2_wt,
            })
            print(f"  Stack {s}: hash=0x{stack_hash:08X}  fc0_bias[0]={fc0_bias[0]}")

        remaining = f.read()
        print(f"Bytes remaining after stacks: {len(remaining)}")
        return ft_biases, ft_weights, psqt_weights, stacks

# ------------------------------------------------------------------
# Compute forward pass from l0_in
# ------------------------------------------------------------------
def fc0_from_l0in(l0_in, stack):
    bias = stack['fc0_bias']
    wt   = stack['fc0_wt']
    fc0  = list(bias)
    for o in range(L0_SIZE):
        for i in range(L0_INPUT):
            fc0[o] += int(l0_in[i]) * wt[o * L0_INPUT + i]
    return fc0

def dual_act(fc0_raw):
    WEIGHT_SHIFT = 6
    SQR_SHIFT    = 7
    fc1_in = [0] * L1_PADDED
    for o in range(15):
        v = fc0_raw[o] >> WEIGHT_SHIFT
        if v < 0: v = 0
        if v > 127: v = 127
        sq = (v * v) >> SQR_SHIFT
        if sq > 127: sq = 127
        fc1_in[o]      = sq   # SqrCReLU
        fc1_in[15 + o] = v    # CReLU
    return fc1_in

def fc1_forward(fc1_in, stack):
    bias = stack['fc1_bias']
    wt   = stack['fc1_wt']
    raw  = list(bias)
    for o in range(L1_PADDED):
        for i in range(L1_PADDED):
            raw[o] += int(fc1_in[i]) * wt[o * L1_PADDED + i]
    return raw

def fc1_act(fc1_raw):
    WEIGHT_SHIFT = 6
    out = []
    for v in fc1_raw:
        v >>= WEIGHT_SHIFT
        if v < 0: v = 0
        if v > 127: v = 127
        out.append(v)
    return out

def fc2_forward(fc2_in, stack):
    bias = stack['fc2_bias']
    wt   = stack['fc2_wt']
    out  = bias
    for i in range(L2_PADDED):
        out += int(fc2_in[i]) * wt[i]
    return out

# ------------------------------------------------------------------
# Starting position feature computation (EXchess square numbering: a1=0)
# ------------------------------------------------------------------
PS_NB  = 704   # 11 * 64
KING_BUCKETS_WHITE = [  # indexed by ksq_f (= ksq for white, ksq^56 for black)
    28, 29, 30, 31, 31, 30, 29, 28,
    24, 25, 26, 27, 27, 26, 25, 24,
    20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16,
    12, 13, 14, 15, 15, 14, 13, 12,
     8,  9, 10, 11, 11, 10,  9,  8,
     4,  5,  6,  7,  7,  6,  5,  4,
     0,  1,  2,  3,  3,  2,  1,  0,
]

def make_feature(persp, ksq, psq, ptype, pside):
    """EXchess halfkav2_feature — returns feature index or -1."""
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
    WHITE, BLACK = 1, 0
    if ptype == KING and pside == persp:
        return -1
    flip   = 56 if persp == BLACK else 0
    ksq_f  = ksq ^ flip
    orient = 7 if (ksq_f & 7) < 4 else 0
    psq_o  = (psq ^ flip) ^ orient
    bucket = KING_BUCKETS_WHITE[ksq_f] * PS_NB
    if ptype == KING:
        ps = 640
    else:
        is_own = (pside == persp)
        ps = (ptype - 1) * 128 + (0 if is_own else 64)
    return bucket + ps + psq_o

def starting_pos_features(persp):
    """Return list of active feature indices for the starting position."""
    WHITE, BLACK = 1, 0
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
    pieces = [
        # (sq, ptype, pside)
        (4,  KING,   WHITE),  # own king — will be filtered
        (60, KING,   BLACK),
        # White pawns a2-h2
        *[(8+i, PAWN, WHITE) for i in range(8)],
        # Black pawns a7-h7
        *[(48+i, PAWN, BLACK) for i in range(8)],
        # White pieces on rank 1
        (0, ROOK, WHITE), (7, ROOK, WHITE),
        (1, KNIGHT, WHITE), (6, KNIGHT, WHITE),
        (2, BISHOP, WHITE), (5, BISHOP, WHITE),
        (3, QUEEN, WHITE),
        # Black pieces on rank 8
        (56, ROOK, BLACK), (63, ROOK, BLACK),
        (57, KNIGHT, BLACK), (62, KNIGHT, BLACK),
        (58, BISHOP, BLACK), (61, BISHOP, BLACK),
        (59, QUEEN, BLACK),
    ]
    wksq = 4   # e1
    bksq = 60  # e8
    ksq  = wksq if persp == WHITE else bksq
    feats = []
    for sq, pt, sd in pieces:
        fi = make_feature(persp, ksq, sq, pt, sd)
        if fi >= 0:
            feats.append(fi)
    return feats

def verify_accumulator(ft_biases, ft_weights, exchess_acc, persp_name, persp):
    print(f"\nVerifying {persp_name} accumulator for starting position...")
    feats = starting_pos_features(persp)
    print(f"  Active features: {sorted(feats)}")
    acc = list(ft_biases)
    for fi in feats:
        base = fi * HALF_DIMS
        for j in range(HALF_DIMS):
            acc[j] += ft_weights[base + j]
    print(f"  Computed acc[0..15]: {acc[:16]}")
    print(f"  EXchess  acc[0..15]: {exchess_acc[:16]}")
    match = acc[:16] == list(exchess_acc[:16])
    print(f"  First 16 match: {match}")
    if not match:
        print("  Differences:")
        for i, (c, e) in enumerate(zip(acc[:16], exchess_acc[:16])):
            if c != e:
                print(f"    [{i}] computed={c}  exchess={e}")
    return acc

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def sqr_crelu(acc_full):
    """Apply SqrCReLU to a full 1024-element accumulator → 512 int8 values."""
    out = [0] * 512
    for i in range(512):
        va = max(0, min(127, acc_full[i]))
        vb = max(0, min(127, acc_full[i + 512]))
        sq = (va * vb) >> 7
        out[i] = min(127, sq)
    return out

def build_accumulator(ft_biases, ft_weights, pieces, persp):
    """
    Compute accumulator for a given position from scratch.
    pieces = list of (sq, ptype, pside) tuples.
    persp  = 0 (BLACK) or 1 (WHITE).
    Returns 1024-element int list.
    """
    WHITE, BLACK = 1, 0
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
    # Find own king square
    ksq = next(sq for sq, pt, sd in pieces if pt == KING and sd == persp)
    acc = list(ft_biases)
    for sq, pt, sd in pieces:
        fi = make_feature(persp, ksq, sq, pt, sd)
        if fi >= 0:
            base = fi * HALF_DIMS
            for j in range(HALF_DIMS):
                acc[j] += ft_weights[base + j]
    return acc

def run_full_eval(ft_biases, ft_weights, stacks, pieces_stm, pieces_opp,
                  stm, piece_count, label=""):
    """
    Compute NNUE eval from scratch for a position.
    pieces_stm / pieces_opp: lists of (sq, ptype, pside) for both sides combined.
    stm = 1 (WHITE) or 0 (BLACK).
    """
    all_pieces = pieces_stm  # should include all pieces from both sides
    acc_stm = build_accumulator(ft_biases, ft_weights, all_pieces, stm)
    acc_opp = build_accumulator(ft_biases, ft_weights, all_pieces, stm ^ 1)

    stack_idx = max(0, min(7, (piece_count - 1) // 4))
    sk = stacks[stack_idx]

    l0_stm = sqr_crelu(acc_stm)
    l0_opp = sqr_crelu(acc_opp)
    l0_in  = l0_stm + l0_opp

    fc0_raw = fc0_from_l0in(l0_in, sk)
    fc1_in  = dual_act(fc0_raw)
    fc1_raw = fc1_forward(fc1_in, sk)
    fc2_in  = fc1_act(fc1_raw)
    fc2_out = fc2_forward(fc2_in, sk)
    fwd_out = int(fc0_raw[15] * 9600 / 8128)
    positional = fc2_out + fwd_out

    # PSQT: compute per-bucket psqt for stm and opp
    WHITE, BLACK = 1, 0
    KING = 6
    ksq_stm = next(sq for sq, pt, sd in all_pieces if pt == KING and sd == stm)
    ksq_opp = next(sq for sq, pt, sd in all_pieces if pt == KING and sd == (stm^1))

    # PSQT: sum psqt_weights[fi * PSQT_BKTS + stack_idx] for each active feature
    psqt_stm = 0
    psqt_opp = 0
    for sq, pt, sd in all_pieces:
        fi_stm = make_feature(stm,     ksq_stm, sq, pt, sd)
        fi_opp = make_feature(stm ^ 1, ksq_opp, sq, pt, sd)
        if fi_stm >= 0:
            psqt_stm += psqt_weights[fi_stm * PSQT_BKTS + stack_idx]
        if fi_opp >= 0:
            psqt_opp += psqt_weights[fi_opp * PSQT_BKTS + stack_idx]
    psqt_diff = psqt_stm - psqt_opp  # raw int32 diff (before any scaling)

    if label:
        print(f"\n{'='*60}\n{label} (stack={stack_idx}, pc={piece_count})")
    print(f"  acc[stm][0..15]:  {acc_stm[:16]}")
    print(f"  acc[opp][0..15]:  {acc_opp[:16]}")
    print(f"  l0_in[0..15]:     {l0_in[:16]}")
    print(f"  fc0_raw:          {fc0_raw}")
    print(f"  fc1_in:           {fc1_in}")
    print(f"  fc2_in:           {fc2_in}")
    print(f"  fc2_out={fc2_out}  fwd={fwd_out}  positional={positional}")
    print(f"  psqt_stm={psqt_stm}  psqt_opp={psqt_opp}  psqt_diff={psqt_diff}")
    print(f"  positional/57.76 = {positional/57.76:.1f} cp")
    print(f"  psqt_diff/2/57.76 = {(psqt_diff/2)/57.76:.1f} cp")
    print(f"  TOTAL (SF scale) = {(psqt_diff/2 + positional)/57.76:.1f} cp")
    return positional, acc_stm, fc0_raw

if __name__ == '__main__':
    print("=" * 60)
    ft_biases, ft_weights, psqt_weights, stacks = parse_net(NET_PATH)
    s7 = stacks[7]
    s1 = stacks[1]

    # ---------------------------------------------------------------
    # Position 1: starting pos minus white h1 rook  (bucket=7, 31 pcs)
    # rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq
    # ---------------------------------------------------------------
    PAWN,KNIGHT,BISHOP,ROOK,QUEEN,KING = 1,2,3,4,5,6
    WHITE,BLACK = 1,0
    pos1 = [
        (4, KING, WHITE), (60, KING, BLACK),
        *[(8+i, PAWN, WHITE) for i in range(8)],
        *[(48+i, PAWN, BLACK) for i in range(8)],
        (0, ROOK, WHITE),                          # a1 rook (h1 removed)
        (1, KNIGHT, WHITE), (6, KNIGHT, WHITE),
        (2, BISHOP, WHITE), (5, BISHOP, WHITE),
        (3, QUEEN, WHITE),
        (56, ROOK, BLACK), (63, ROOK, BLACK),
        (57, KNIGHT, BLACK), (62, KNIGHT, BLACK),
        (58, BISHOP, BLACK), (61, BISHOP, BLACK),
        (59, QUEEN, BLACK),
    ]
    pos_fc1, acc1, fc0_1 = run_full_eval(ft_biases, ft_weights, stacks,
                                          pos1, pos1, WHITE, 31,
                                          "Position 1 (missing h1 rook, bucket=7)")

    # ---------------------------------------------------------------
    # Position 3: kn6/p5PP/8/8/KN6/8/8/8 w  (bucket=1, 7 pcs)
    # ---------------------------------------------------------------
    pos3 = [
        (24, KING, WHITE), (57, KING, BLACK),   # a4=24, b8=57
        (25, KNIGHT, WHITE),                     # b4=25
        (54, PAWN, WHITE), (55, PAWN, WHITE),   # g7=54, h7=55
        (48, PAWN, BLACK),                       # a7=48
        (58, KNIGHT, BLACK),                     # c8=58 — wait let me recheck
    ]
    # kn6/p5PP/8/8/KN6/8/8/8:
    # Rank8: k=a8=56, n=b8=57
    # Rank7: p=a7=48, P=g7=54, P=h7=55
    # Rank4: K=a4=24, N=b4=25
    pos3 = [
        (24, KING, WHITE), (56, KING, BLACK),
        (25, KNIGHT, WHITE),
        (54, PAWN, WHITE), (55, PAWN, WHITE),
        (48, PAWN, BLACK),
        (57, KNIGHT, BLACK),
    ]
    pos_fc3, acc3, fc0_3 = run_full_eval(ft_biases, ft_weights, stacks,
                                          pos3, pos3, WHITE, 7,
                                          "Position 3 (endgame, bucket=1)")

    # ---------------------------------------------------------------
    # Compare accumulator saturation
    # ---------------------------------------------------------------
    def saturation_report(acc, label):
        n_gt127 = sum(1 for v in acc if v > 127)
        n_lt0   = sum(1 for v in acc if v < 0)
        n_in    = 1024 - n_gt127 - n_lt0
        print(f"  {label}: >127={n_gt127}, <0={n_lt0}, in[0,127]={n_in}  "
              f"(SqrCReLU zero-out: {n_lt0}  saturated: {n_gt127})")

    print("\n" + "="*60)
    print("Accumulator saturation analysis:")
    saturation_report(acc1, "Pos1 stm(WHITE)")
    saturation_report(acc3, "Pos3 stm(WHITE)")

    # ---------------------------------------------------------------
    # l0_in from NNUE_DEBUG_VERBOSE (optional, for EXchess comparison)
    # ---------------------------------------------------------------
    l0_in_exchess = [0] * L0_INPUT
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as fh:
            for tok in fh.read().split():
                if '=' in tok:
                    idx, val = tok.split('=')
                    l0_in_exchess[int(idx.strip('[]'))] = int(val)
        print("\nEXchess l0_in provided — computing FC against stack 7:")
        fc0_raw_ex = fc0_from_l0in(l0_in_exchess, s7)
        fc1_in_ex  = dual_act(fc0_raw_ex)
        fc1_raw_ex = fc1_forward(fc1_in_ex, s7)
        fc2_in_ex  = fc1_act(fc1_raw_ex)
        fc2_out_ex = fc2_forward(fc2_in_ex, s7)
        fwd_ex     = int(fc0_raw_ex[15] * 9600 / 8128)
        print(f"  fc0_raw: {fc0_raw_ex}")
        print(f"  fc2={fc2_out_ex}  fwd={fwd_ex}  positional={fc2_out_ex+fwd_ex}")
