// EXchess NNUE evaluation — HalfKAv2_hm format (Stockfish 15/16 era)
// Written from scratch; Stockfish source consulted only for file-format layout.
//
// File format confirmed from nn-ad9b42354671.nnue (Stockfish 15.1 exact release):
//   [Header] version(4) + hash(4) + desc_size(4) + desc(N)
//   [FT]     ft_hash(4) + LEB128(biases 1024 i16) + LEB128(weights 22528×1024 i16)
//             + LEB128(psqt 22528×8 i32)
//   [Stacks] 8 × [stack_hash(4) + FC0_bias(16×i32) + FC0_wt(16×1024×i8)
//                               + FC1_bias(32×i32) + FC1_wt(32×32×i8)
//                               + FC2_bias(1×i32)  + FC2_wt(32×i8)]
//   Each stack = 4+64+16384+128+1024+4+32 = 17640 bytes
//   (no separate net_hash; the FT section ends immediately before stack 0's hash)

#include <stdint.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include "define.h"
#include "chess.h"
#include "nnue.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

bool nnue_available = false;

// ---------------------------------------------------------------------------
// Weight storage
// ---------------------------------------------------------------------------
static int16_t *ft_biases    = nullptr; // [NNUE_HALF_DIMS]
static int16_t *ft_weights   = nullptr; // [NNUE_FT_INPUTS × NNUE_HALF_DIMS]
static int32_t *psqt_weights = nullptr; // [NNUE_FT_INPUTS × NNUE_PSQT_BKTS]

// FC0: output-major weight layout, 16 outputs × 3072 inputs (vdotq-reordered at load)
static int32_t l0_biases [NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static int8_t  l0_weights[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT];

// FC1: output-major, 32 outputs × 32 padded-inputs
static int32_t l1_biases [NNUE_LAYER_STACKS][NNUE_L1_SIZE];
static int8_t  l1_weights[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED];

// FC2 (output): 1 output × 32 padded-inputs
static int32_t out_biases [NNUE_LAYER_STACKS];
static int8_t  out_weights[NNUE_LAYER_STACKS][NNUE_L2_PADDED];

// ---------------------------------------------------------------------------
// HalfKAv2_hm lookup tables
// ---------------------------------------------------------------------------
static const int PS_NB = 704;  // 11 piece-type slots × 64 squares

static const int KingBuckets[64] = {
    28*PS_NB, 29*PS_NB, 30*PS_NB, 31*PS_NB, 31*PS_NB, 30*PS_NB, 29*PS_NB, 28*PS_NB,
    24*PS_NB, 25*PS_NB, 26*PS_NB, 27*PS_NB, 27*PS_NB, 26*PS_NB, 25*PS_NB, 24*PS_NB,
    20*PS_NB, 21*PS_NB, 22*PS_NB, 23*PS_NB, 23*PS_NB, 22*PS_NB, 21*PS_NB, 20*PS_NB,
    16*PS_NB, 17*PS_NB, 18*PS_NB, 19*PS_NB, 19*PS_NB, 18*PS_NB, 17*PS_NB, 16*PS_NB,
    12*PS_NB, 13*PS_NB, 14*PS_NB, 15*PS_NB, 15*PS_NB, 14*PS_NB, 13*PS_NB, 12*PS_NB,
     8*PS_NB,  9*PS_NB, 10*PS_NB, 11*PS_NB, 11*PS_NB, 10*PS_NB,  9*PS_NB,  8*PS_NB,
     4*PS_NB,  5*PS_NB,  6*PS_NB,  7*PS_NB,  7*PS_NB,  6*PS_NB,  5*PS_NB,  4*PS_NB,
     0*PS_NB,  1*PS_NB,  2*PS_NB,  3*PS_NB,  3*PS_NB,  2*PS_NB,  1*PS_NB,  0*PS_NB,
};

// ---------------------------------------------------------------------------
// HalfKAv2_hm feature index
// ---------------------------------------------------------------------------
static inline int halfkav2_feature(int persp, int ksq,
                                   int psq,  int ptype, int pside)
{
    // Note: own king IS included as a feature (same PS_KING slot as enemy king).
    // Both kings map to PS_KING=640; the own king's feature index is valid and
    // contributes to the accumulator.  Only excluded: no-piece (ptype == 0).

    int flip   = (persp == BLACK) ? 56 : 0;
    int ksq_f  = ksq ^ flip;
    // Horizontal mirror: normalize so the own king is always on the right half.
    // Flip file when king is on files 0-3 (a-d = queen side) for BOTH perspectives.
    // This matches Stockfish HalfKAv2_hm: orient = s ^ (king_file<4 ? 7 : 0) ^ rank_flip.
    // File is unchanged by the rank-flip, so (ksq_f & 7) == (ksq & 7).
    int orient = ((ksq_f & 7) < 4) ? 7 : 0;
    int psq_o  = (psq ^ flip) ^ orient;
    int bucket = KingBuckets[ksq_f];

    int ps;
    if (ptype == KING) {
        ps = 640;
    } else {
        bool is_own = (pside == persp);
        ps = (ptype - 1) * 128 + (is_own ? 0 : 64);
    }

    return bucket + ps + psq_o;
}

// ---------------------------------------------------------------------------
// Accumulator add/sub helpers
// ---------------------------------------------------------------------------
static inline void add_feat(int16_t *half_acc, int32_t *half_psqt, int fidx)
{
    if (fidx < 0) return;
    const int16_t *w  = ft_weights   + (size_t)fidx * NNUE_HALF_DIMS;
    const int32_t *pw = psqt_weights + (size_t)fidx * NNUE_PSQT_BKTS;
    for (int j = 0; j < NNUE_HALF_DIMS; j++) half_acc[j]  += w[j];
    for (int b = 0; b < NNUE_PSQT_BKTS; b++) half_psqt[b] += pw[b];
}
static inline void sub_feat(int16_t *half_acc, int32_t *half_psqt, int fidx)
{
    if (fidx < 0) return;
    const int16_t *w  = ft_weights   + (size_t)fidx * NNUE_HALF_DIMS;
    const int32_t *pw = psqt_weights + (size_t)fidx * NNUE_PSQT_BKTS;
    for (int j = 0; j < NNUE_HALF_DIMS; j++) half_acc[j]  -= w[j];
    for (int b = 0; b < NNUE_PSQT_BKTS; b++) half_psqt[b] -= pw[b];
}

// ---------------------------------------------------------------------------
// LEB128 decompressor (used for FT biases, weights, and PSQT weights)
// ---------------------------------------------------------------------------
static bool read_leb128_i16(FILE *f, int16_t *buf, size_t count)
{
    char magic[18] = {};
    if (fread(magic, 1, 17, f) != 17) return false;

    if (memcmp(magic, "COMPRESSED_LEB128", 17) != 0) {
        fseek(f, -17, SEEK_CUR);
        return fread(buf, sizeof(int16_t), count, f) == count;
    }

    uint32_t nbytes;
    if (fread(&nbytes, 4, 1, f) != 1) return false;

    unsigned char *cbuf = (unsigned char*)malloc(nbytes);
    if (!cbuf) return false;
    if (fread(cbuf, 1, nbytes, f) != nbytes) { free(cbuf); return false; }

    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        int32_t val = 0; int shift = 0;
        unsigned char byte;
        do {
            if (pos >= nbytes) { free(cbuf); return false; }
            byte = cbuf[pos++];
            val |= (int32_t)(byte & 0x7F) << shift;
            shift += 7;
        } while (byte & 0x80);
        if (shift < 32 && (byte & 0x40))
            val |= ~((int32_t)0) << shift;
        buf[i] = (int16_t)val;
    }
    free(cbuf);
    return true;
}

static bool read_leb128_i32(FILE *f, int32_t *buf, size_t count)
{
    char magic[18] = {};
    if (fread(magic, 1, 17, f) != 17) return false;

    if (memcmp(magic, "COMPRESSED_LEB128", 17) != 0) {
        fseek(f, -17, SEEK_CUR);
        return fread(buf, sizeof(int32_t), count, f) == count;
    }

    uint32_t nbytes;
    if (fread(&nbytes, 4, 1, f) != 1) return false;

    unsigned char *cbuf = (unsigned char*)malloc(nbytes);
    if (!cbuf) return false;
    if (fread(cbuf, 1, nbytes, f) != nbytes) { free(cbuf); return false; }

    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        int64_t val = 0; int shift = 0;
        unsigned char byte;
        do {
            if (pos >= nbytes) { free(cbuf); return false; }
            byte = cbuf[pos++];
            val |= (int64_t)(byte & 0x7F) << shift;
            shift += 7;
        } while (byte & 0x80);
        if (shift < 64 && (byte & 0x40))
            val |= ~((int64_t)0) << shift;
        buf[i] = (int32_t)val;
    }
    free(cbuf);
    return true;
}

static uint32_t read_u32(FILE *f) {
    uint32_t v = 0;
    fread(&v, 4, 1, f);
    return v;
}

// ---------------------------------------------------------------------------
// nnue_load
// ---------------------------------------------------------------------------
bool nnue_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("NNUE: could not open %s\n", path);
        return false;
    }

    // Header
    uint32_t version   = read_u32(f);
    uint32_t file_hash = read_u32(f);
    uint32_t desc_size = read_u32(f);
    printf("NNUE: version=0x%08X  hash=0x%08X  desc_size=%u\n",
           version, file_hash, desc_size);

    if (desc_size > 0) {
        char *desc = new char[desc_size + 1];
        size_t nr  = fread(desc, 1, desc_size, f);
        desc[nr]   = '\0';
        printf("NNUE: architecture: %s\n", desc);
        delete[] desc;
        if (nr != desc_size) { fclose(f); return false; }
    }

    // Feature Transformer
    uint32_t ft_hash = read_u32(f);
    printf("NNUE: ft_hash=0x%08X\n", ft_hash);

    if (!ft_biases)    ft_biases    = new int16_t[NNUE_HALF_DIMS];
    if (!ft_weights)   ft_weights   = new int16_t[(size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS];
    if (!psqt_weights) psqt_weights = new int32_t[(size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS];

    printf("NNUE: reading FT biases [%d int16] ...\n", NNUE_HALF_DIMS);
    if (!read_leb128_i16(f, ft_biases, NNUE_HALF_DIMS)) {
        printf("NNUE: FT bias read failed\n"); fclose(f); return false;
    }

    size_t ft_w = (size_t)NNUE_FT_INPUTS * NNUE_HALF_DIMS;
    printf("NNUE: reading FT weights [%zu int16] ...\n", ft_w);
    if (!read_leb128_i16(f, ft_weights, ft_w)) {
        printf("NNUE: FT weight read failed\n"); fclose(f); return false;
    }

    size_t psqt_w = (size_t)NNUE_FT_INPUTS * NNUE_PSQT_BKTS;
    printf("NNUE: reading PSQT weights [%zu int32] ...\n", psqt_w);
    if (!read_leb128_i32(f, psqt_weights, psqt_w)) {
        printf("NNUE: PSQT weight read failed\n"); fclose(f); return false;
    }
    printf("NNUE: FT + PSQT loaded OK\n");

    // Network: 8 layer stacks.
    // Each stack begins with a 4-byte hash (no separate net_hash before the stacks).
    // Layout per stack:
    //   stack_hash(4) + FC0_bias(16*4) + FC0_wt(16*3072) +
    //   FC1_bias(32*4) + FC1_wt(32*32) + FC2_bias(4) + FC2_wt(32)
    //   = 4 + 64 + 49152 + 128 + 1024 + 4 + 32 = 50408 bytes
    printf("NNUE: reading %d layer stacks ...\n", NNUE_LAYER_STACKS);

    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        uint32_t stack_hash = read_u32(f);
        (void)stack_hash;  // hash value not verified

        // FC0
        if (fread(l0_biases[s], sizeof(int32_t), NNUE_L0_SIZE, f) != (size_t)NNUE_L0_SIZE)
            { printf("NNUE: stack %d FC0 bias read failed\n", s); fclose(f); return false; }
        size_t fc0_w = (size_t)NNUE_L0_SIZE * NNUE_L0_INPUT;
        {
            // Read output-major [o * NNUE_L0_INPUT + i] into a temp buffer,
            // then rearrange into the vdotq-friendly layout:
            //   For each 4-input block ib (0..255) and 4-output block ob (0..3),
            //   store 16 bytes as [w(i0,o0),w(i1,o0),w(i2,o0),w(i3,o0),
            //                      w(i0,o1),w(i1,o1),w(i2,o1),w(i3,o1), ...]
            //   i.e. l0_weights[s][ib*64 + ob*16 + k*4 + j]
            //      = original[o = ob*4+k][i = ib*4+j]
            // This lets a single vdotq_s32 call accumulate 4 inputs into 4 outputs.
            int8_t *tmp = new int8_t[fc0_w];
            if (fread(tmp, sizeof(int8_t), fc0_w, f) != fc0_w)
                { delete[] tmp; printf("NNUE: stack %d FC0 weight read failed\n", s); fclose(f); return false; }
            for (int o = 0; o < NNUE_L0_SIZE; o++) {
                int ob = o / 4, k = o % 4;
                for (int i = 0; i < NNUE_L0_INPUT; i++) {
                    int ib = i / 4, j = i % 4;
                    l0_weights[s][ib * 64 + ob * 16 + k * 4 + j] =
                        tmp[o * NNUE_L0_INPUT + i];
                }
            }
            delete[] tmp;
        }

        // FC1
        if (fread(l1_biases[s], sizeof(int32_t), NNUE_L1_SIZE, f) != (size_t)NNUE_L1_SIZE)
            { printf("NNUE: stack %d FC1 bias read failed\n", s); fclose(f); return false; }
        size_t fc1_w = (size_t)NNUE_L1_SIZE * NNUE_L1_PADDED;
        {
            // Read output-major [o * NNUE_L1_PADDED + i] into a temp buffer,
            // then rearrange into the vdotq-friendly layout (same scheme as FC0):
            //   l1_weights[s][ib*128 + ob*16 + k*4 + j]
            //     = original[o = ob*4+k][i = ib*4+j]
            //   ib=i/4, j=i%4, ob=o/4, k=o%4  (i∈[0,31], o∈[0,31])
            int8_t tmp[NNUE_L1_SIZE * NNUE_L1_PADDED];
            if (fread(tmp, sizeof(int8_t), fc1_w, f) != fc1_w)
                { printf("NNUE: stack %d FC1 weight read failed\n", s); fclose(f); return false; }
#ifdef __ARM_FEATURE_DOTPROD
            for (int o = 0; o < NNUE_L1_SIZE; o++) {
                int ob = o / 4, k = o % 4;
                for (int i = 0; i < NNUE_L1_PADDED; i++) {
                    int ib = i / 4, j = i % 4;
                    l1_weights[s][ib * 128 + ob * 16 + k * 4 + j] = tmp[o * NNUE_L1_PADDED + i];
                }
            }
#else
            memcpy(l1_weights[s], tmp, fc1_w);
#endif
        }

        // FC2 (output layer)
        if (fread(&out_biases[s], sizeof(int32_t), 1, f) != 1)
            { printf("NNUE: stack %d FC2 bias read failed\n", s); fclose(f); return false; }
        if (fread(out_weights[s], sizeof(int8_t), NNUE_L2_PADDED, f) != (size_t)NNUE_L2_PADDED)
            { printf("NNUE: stack %d FC2 weight read failed\n", s); fclose(f); return false; }
    }

    fclose(f);
    nnue_available = true;
    printf("NNUE: all %d stacks loaded OK\n", NNUE_LAYER_STACKS);
#if TDLEAF
    nnue_init_fp32_weights();
#endif
    return true;
}

// ---------------------------------------------------------------------------
// nnue_init_accumulator
// ---------------------------------------------------------------------------
void nnue_init_accumulator(NNUEAccumulator &acc, const position &pos)
{
    for (int persp = 0; persp < 2; persp++) {
        memcpy(acc.acc[persp], ft_biases, NNUE_HALF_DIMS * sizeof(int16_t));
        memset(acc.psqt[persp], 0, NNUE_PSQT_BKTS * sizeof(int32_t));

        int ksq = pos.plist[persp][KING][1];

        for (int side = 0; side < 2; side++) {
            for (int ptype = PAWN; ptype <= KING; ptype++) {
                for (int i = 1; i <= pos.plist[side][ptype][0]; i++) {
                    int psq  = pos.plist[side][ptype][i];
                    int fidx = halfkav2_feature(persp, ksq, psq, ptype, side);
                    add_feat(acc.acc[persp], acc.psqt[persp], fidx);
                }
            }
        }

        acc.dirty[persp] = false;
    }
    acc.computed = true;
}

// ---------------------------------------------------------------------------
// nnue_update_accumulator
// ---------------------------------------------------------------------------
void nnue_update_accumulator(NNUEAccumulator &acc,
                             const position  &before,
                             const position  &after,
                             move mv)
{
    int from     = mv.b.from;
    int to       = mv.b.to;
    int mtype    = mv.b.type;
    int mover_pt = PTYPE(before.sq[from]);
    int mover_sd = PSIDE(before.sq[from]);
    int capt_pt  = PTYPE(before.sq[to]);

    int wksq = after.plist[WHITE][KING][1];
    int bksq = after.plist[BLACK][KING][1];

    if (mover_pt == KING) {
        acc.dirty[mover_sd] = true;

        int opp_sd   = mover_sd ^ 1;
        int opp_king = (opp_sd == WHITE) ? wksq : bksq;

        if (!acc.dirty[opp_sd]) {
            sub_feat(acc.acc[opp_sd], acc.psqt[opp_sd],
                     halfkav2_feature(opp_sd, opp_king, from, KING, mover_sd));
            add_feat(acc.acc[opp_sd], acc.psqt[opp_sd],
                     halfkav2_feature(opp_sd, opp_king, to,   KING, mover_sd));

            // King capture: remove the captured piece from the opponent's accumulator.
            if (capt_pt) {
                sub_feat(acc.acc[opp_sd], acc.psqt[opp_sd],
                         halfkav2_feature(opp_sd, opp_king, to, capt_pt, mover_sd ^ 1));
            }

            if (mtype & CASTLE) {
                int rook_from, rook_to;
                if (mover_sd == WHITE) {
                    if (to == 6) { rook_from = before.Krook[WHITE]; rook_to = 5; }
                    else         { rook_from = before.Qrook[WHITE]; rook_to = 3; }
                } else {
                    if (to == 62) { rook_from = before.Krook[BLACK]; rook_to = 61; }
                    else          { rook_from = before.Qrook[BLACK]; rook_to = 59; }
                }
                sub_feat(acc.acc[opp_sd], acc.psqt[opp_sd],
                         halfkav2_feature(opp_sd, opp_king, rook_from, ROOK, mover_sd));
                add_feat(acc.acc[opp_sd], acc.psqt[opp_sd],
                         halfkav2_feature(opp_sd, opp_king, rook_to,   ROOK, mover_sd));
            }
        }
        return;
    }

    for (int persp = 0; persp < 2; persp++) {
        if (acc.dirty[persp]) continue;

        int our_king = (persp == WHITE) ? wksq : bksq;

        sub_feat(acc.acc[persp], acc.psqt[persp],
                 halfkav2_feature(persp, our_king, from, mover_pt, mover_sd));

        if (capt_pt && !(mtype & EP)) {
            sub_feat(acc.acc[persp], acc.psqt[persp],
                     halfkav2_feature(persp, our_king, to, capt_pt, mover_sd ^ 1));
        }

        if (mtype & EP) {
            int ep_sq = mover_sd ? (to - 8) : (to + 8);
            sub_feat(acc.acc[persp], acc.psqt[persp],
                     halfkav2_feature(persp, our_king, ep_sq, PAWN, mover_sd ^ 1));
        }

        if (mtype & PROMOTE) {
            add_feat(acc.acc[persp], acc.psqt[persp],
                     halfkav2_feature(persp, our_king, to, mv.b.promote, mover_sd));
        } else {
            add_feat(acc.acc[persp], acc.psqt[persp],
                     halfkav2_feature(persp, our_king, to, mover_pt, mover_sd));
        }
    }
}

// ---------------------------------------------------------------------------
// nnue_record_delta — record feature-index changes for a move (no ft_weights access)
// ---------------------------------------------------------------------------
void nnue_record_delta(NNUEAccumulator &acc,
                       const position  &before,
                       const position  &after,
                       move mv)
{
    acc.computed        = false;
    acc.n_add[0]        = acc.n_add[1] = 0;
    acc.n_sub[0]        = acc.n_sub[1] = 0;
    acc.need_refresh[0] = acc.need_refresh[1] = false;

    int from     = mv.b.from;
    int to       = mv.b.to;
    int mtype    = mv.b.type;
    int mover_pt = PTYPE(before.sq[from]);
    int mover_sd = PSIDE(before.sq[from]);
    int capt_pt  = PTYPE(before.sq[to]);

    int wksq = after.plist[WHITE][KING][1];
    int bksq = after.plist[BLACK][KING][1];

    if (mover_pt == KING) {
        // Moving king's own perspective must be fully rebuilt.
        acc.need_refresh[mover_sd] = true;

        // Opponent's perspective: incremental update for king relocation.
        int opp_sd   = mover_sd ^ 1;
        int opp_king = (opp_sd == WHITE) ? wksq : bksq;

        int fi;
        fi = halfkav2_feature(opp_sd, opp_king, from, KING, mover_sd);
        if (fi >= 0) acc.sub[opp_sd][acc.n_sub[opp_sd]++] = fi;
        fi = halfkav2_feature(opp_sd, opp_king, to,   KING, mover_sd);
        if (fi >= 0) acc.add[opp_sd][acc.n_add[opp_sd]++] = fi;

        // King capture: remove the captured piece from the opponent's accumulator.
        if (capt_pt) {
            fi = halfkav2_feature(opp_sd, opp_king, to, capt_pt, mover_sd ^ 1);
            if (fi >= 0) acc.sub[opp_sd][acc.n_sub[opp_sd]++] = fi;
        }

        if (mtype & CASTLE) {
            int rook_from, rook_to;
            if (mover_sd == WHITE) {
                if (to == 6) { rook_from = before.Krook[WHITE]; rook_to = 5; }
                else         { rook_from = before.Qrook[WHITE]; rook_to = 3; }
            } else {
                if (to == 62) { rook_from = before.Krook[BLACK]; rook_to = 61; }
                else          { rook_from = before.Qrook[BLACK]; rook_to = 59; }
            }
            fi = halfkav2_feature(opp_sd, opp_king, rook_from, ROOK, mover_sd);
            if (fi >= 0) acc.sub[opp_sd][acc.n_sub[opp_sd]++] = fi;
            fi = halfkav2_feature(opp_sd, opp_king, rook_to,   ROOK, mover_sd);
            if (fi >= 0) acc.add[opp_sd][acc.n_add[opp_sd]++] = fi;
        }
        return;
    }

    // Non-king move: both perspectives updated incrementally.
    for (int persp = 0; persp < 2; persp++) {
        int our_king = (persp == WHITE) ? wksq : bksq;
        int fi;

        fi = halfkav2_feature(persp, our_king, from, mover_pt, mover_sd);
        if (fi >= 0) acc.sub[persp][acc.n_sub[persp]++] = fi;

        if (capt_pt && !(mtype & EP)) {
            fi = halfkav2_feature(persp, our_king, to, capt_pt, mover_sd ^ 1);
            if (fi >= 0) acc.sub[persp][acc.n_sub[persp]++] = fi;
        }

        if (mtype & EP) {
            int ep_sq = mover_sd ? (to - 8) : (to + 8);
            fi = halfkav2_feature(persp, our_king, ep_sq, PAWN, mover_sd ^ 1);
            if (fi >= 0) acc.sub[persp][acc.n_sub[persp]++] = fi;
        }

        if (mtype & PROMOTE) {
            fi = halfkav2_feature(persp, our_king, to, mv.b.promote, mover_sd);
            if (fi >= 0) acc.add[persp][acc.n_add[persp]++] = fi;
        } else {
            fi = halfkav2_feature(persp, our_king, to, mover_pt, mover_sd);
            if (fi >= 0) acc.add[persp][acc.n_add[persp]++] = fi;
        }
    }
}

// ---------------------------------------------------------------------------
// nnue_apply_delta — materialise a lazy accumulator from its parent
// ---------------------------------------------------------------------------
void nnue_apply_delta(NNUEAccumulator &acc,
                      const NNUEAccumulator &parent_acc,
                      const position &pos)
{
    for (int p = 0; p < 2; p++) {
        if (acc.need_refresh[p]) {
            // King moved for this perspective: full rebuild from position.
            int ksq = pos.plist[p][KING][1];
            memcpy(acc.acc[p], ft_biases, NNUE_HALF_DIMS * sizeof(int16_t));
            memset(acc.psqt[p], 0, NNUE_PSQT_BKTS * sizeof(int32_t));
            for (int side = 0; side < 2; side++) {
                for (int ptype = PAWN; ptype <= KING; ptype++) {
                    for (int i = 1; i <= pos.plist[side][ptype][0]; i++) {
                        int psq = pos.plist[side][ptype][i];
                        add_feat(acc.acc[p], acc.psqt[p],
                                 halfkav2_feature(p, ksq, psq, ptype, side));
                    }
                }
            }
        } else {
            // Incremental: copy parent's accumulator and apply the stored deltas.
            memcpy(acc.acc[p],  parent_acc.acc[p],  NNUE_HALF_DIMS  * sizeof(int16_t));
            memcpy(acc.psqt[p], parent_acc.psqt[p], NNUE_PSQT_BKTS * sizeof(int32_t));
            for (int k = 0; k < acc.n_sub[p]; k++)
                sub_feat(acc.acc[p], acc.psqt[p], acc.sub[p][k]);
            for (int k = 0; k < acc.n_add[p]; k++)
                add_feat(acc.acc[p], acc.psqt[p], acc.add[p][k]);
        }
    }
    acc.dirty[0] = acc.dirty[1] = false;
    acc.computed = true;

#ifdef NNUE_CHECK_LAZY
    // Diagnostic: verify lazy result against full-refresh
    {
        NNUEAccumulator ref;
        ref.dirty[0] = ref.dirty[1] = true;
        nnue_init_accumulator(ref, pos);
        for (int p = 0; p < 2; p++) {
            for (int j = 0; j < NNUE_HALF_DIMS; j++) {
                if (acc.acc[p][j] != ref.acc[p][j]) {
                    fprintf(stderr, "LAZY_MISMATCH persp=%d j=%d lazy=%d ref=%d parent=%d "
                            "n_sub=%d n_add=%d need_refresh=%d\n",
                            p, j, (int)acc.acc[p][j], (int)ref.acc[p][j],
                            (int)parent_acc.acc[p][j],
                            (int)acc.n_sub[p], (int)acc.n_add[p], (int)acc.need_refresh[p]);
                    // Print sub features and their weight at dim j
                    for (int k = 0; k < acc.n_sub[p]; k++) {
                        int fi = acc.sub[p][k];
                        fprintf(stderr, "  SUB[%d] fi=%d w[j]=%d\n",
                                k, fi, (int)ft_weights[(int64_t)fi * NNUE_HALF_DIMS + j]);
                    }
                    for (int k = 0; k < acc.n_add[p]; k++) {
                        int fi = acc.add[p][k];
                        fprintf(stderr, "  ADD[%d] fi=%d w[j]=%d\n",
                                k, fi, (int)ft_weights[(int64_t)fi * NNUE_HALF_DIMS + j]);
                    }
                    // Scan ref features for this perspective and print those whose
                    // weight at dim j is non-zero (to identify the missing/extra feature).
                    {
                        int ksq = pos.plist[p][KING][1];
                        fprintf(stderr, "  Ref features (persp=%d ksq=%d):\n", p, ksq);
                        for (int side = 0; side < 2; side++) {
                            for (int ptype = PAWN; ptype <= KING; ptype++) {
                                for (int i = 1; i <= pos.plist[side][ptype][0]; i++) {
                                    int psq = pos.plist[side][ptype][i];
                                    int fi = halfkav2_feature(p, ksq, psq, ptype, side);
                                    if (fi >= 0) {
                                        int w = (int)ft_weights[(int64_t)fi * NNUE_HALF_DIMS + j];
                                        if (w != 0)
                                            fprintf(stderr, "  feat fi=%d side=%d ptype=%d psq=%d w[j]=%d\n",
                                                    fi, side, ptype, psq, w);
                                    }
                                }
                            }
                        }
                    }
                    goto done_check;
                }
            }
        }
        done_check:;
    }
#endif
}

// ---------------------------------------------------------------------------
// nnue_evaluate — forward pass → score from stm's perspective
//
// Input preparation (SqrCReLU — Stockfish 15.1 exact):
//   For each perspective (stm, opp), from NNUE_HALF_DIMS=1024 int16 accumulator:
//     - Clamp acc[i] and acc[i+512] directly to [0, 127] (NO right-shift)
//     - SqrCReLU on pairs for i=0..511:
//         a = clamp(acc[i],       0, 127)
//         b = clamp(acc[i+512],   0, 127)
//         out[i] = (a*b) >> SQR_SHIFT   (≤ 127, no overflow)
//     - No CReLU part — only 512 values per perspective
//   Total FC0 input: [stm_512 | opp_512] = 1024 int8 values
//
// FC0 (16 outputs, output-major weights 16×1024):
//   raw[o] = bias[o] + dot(l0_in, weights[o*1024 .. (o+1)*1024-1])
//
// Dual activation of FC0 outputs 0..14:
//   v = clamp(raw[o]>>WEIGHT_SHIFT, 0, 127)
//   fc1_in[o]    = clamp((v*v)>>SQR_SHIFT, 0, 127)   // SqrCReLU
//   fc1_in[15+o] = v                                  // CReLU
// FC0 output 15 contributes directly to the final output.
//
// FC1 (32 outputs, 32-padded input; fc1_in[30..31] = 0):
//   raw1[o] = bias[o] + dot(fc1_in[0..31], weights[o*32 .. o*32+31])
//   fc2_in[o] = clamp(raw1[o]>>WEIGHT_SHIFT, 0, 127)
//
// FC2 output + FC0 direct + PSQT → centipawns
// ---------------------------------------------------------------------------
int nnue_evaluate(const NNUEAccumulator &acc, int stm, int piece_count)
{
    if (piece_count < 1)  piece_count = 1;
    if (piece_count > 32) piece_count = 32;
    int stack = (piece_count - 1) / 4;  // 0..7

    // 1. SqrCReLU activation: produce 1024 int8 values
    //    Layout: [stm_sqr(0..511) | opp_sqr(0..511)]
    //    Accumulator int16 clamped directly to [0,127] — no right-shift.
    int8_t l0_in[NNUE_L0_INPUT];
    const int16_t *persp_acc[2] = { acc.acc[stm], acc.acc[stm ^ 1] };

#ifdef __ARM_NEON
    {
        const int16x8_t zero16    = vdupq_n_s16(0);
        const int16x8_t max127_16 = vdupq_n_s16(127);
        for (int p = 0; p < 2; p++) {
            const int16_t *a = persp_acc[p];
            int8_t *out = l0_in + p * 512;
            for (int i = 0; i < 512; i += 8) {
                // Clamp directly to [0,127] — no >> 6 shift
                int16x8_t va_c = vminq_s16(vmaxq_s16(vld1q_s16(a + i),       zero16), max127_16);
                int16x8_t vb_c = vminq_s16(vmaxq_s16(vld1q_s16(a + i + 512), zero16), max127_16);
                int8x8_t  va8  = vmovn_s16(va_c);
                int8x8_t  vb8  = vmovn_s16(vb_c);
                // SqrCReLU: max product = 127*127=16129, >>7 = 126 ≤ 127, no clamp needed
                int8x8_t  sq8  = vmovn_s16(vshrq_n_s16(vmull_s8(va8, vb8), 7)); // SQR_SHIFT=7
                vst1_s8(out + i, sq8); // SqrCReLU → [0..511]
            }
        }
    }
#else
    for (int p = 0; p < 2; p++) {
        const int16_t *a = persp_acc[p];
        int8_t *out = l0_in + p * 512;

        // SqrCReLU on pairs (i, i+512) — clamp directly, no shift
        for (int i = 0; i < 512; i++) {
            int va = (int)a[i];
            int vb = (int)a[i + 512];
            if (va < 0) va = 0; else if (va > 127) va = 127;
            if (vb < 0) vb = 0; else if (vb > 127) vb = 127;
            int sq = (va * vb) >> NNUE_SQR_SHIFT;
            out[i] = (int8_t)(sq > 127 ? 127 : sq);
        }
    }
#endif

    // 2. FC0: 1024 → 16
    //    Weights are stored in vdotq-friendly layout (set at load time):
    //      wt[ib*64 + ob*16 + k*4 + j]  where ib=i/4, j=i%4, ob=o/4, k=o%4
    //    Each loop iteration processes 4 consecutive inputs against all 16 outputs
    //    via 4 vdotq_s32 calls — 256 iterations total (vs 16×256 per-output).
    int32_t fc0_raw[NNUE_L0_SIZE];
    {
        const int32_t *bias = l0_biases[stack];
        const int8_t  *wt   = l0_weights[stack];
#ifdef __ARM_FEATURE_DOTPROD
        // vdotq_s32(acc, b, c): for k=0..3, acc[k] += b[4k+j]*c[4k+j] for j=0..3
        // b = [in[i+0], in[i+1], in[i+2], in[i+3]] repeated 4 times (one per output lane)
        // c (per output block ob) = [w(i+j, o=ob*4+k)] arranged to match b's lane pattern
        int32x4_t acc0 = vld1q_s32(bias);
        int32x4_t acc1 = vld1q_s32(bias + 4);
        int32x4_t acc2 = vld1q_s32(bias + 8);
        int32x4_t acc3 = vld1q_s32(bias + 12);
        for (int ib = 0; ib < NNUE_L0_INPUT / 4; ib++) {
            // Replicate the 4-byte input chunk across all 4 int32 lanes.
            int8x16_t b = vreinterpretq_s8_s32(vdupq_n_s32(*(const int32_t *)(l0_in + ib * 4)));
            acc0 = vdotq_s32(acc0, b, vld1q_s8(wt + ib * 64));      // out block 0 (o 0-3)
            acc1 = vdotq_s32(acc1, b, vld1q_s8(wt + ib * 64 + 16)); // out block 1 (o 4-7)
            acc2 = vdotq_s32(acc2, b, vld1q_s8(wt + ib * 64 + 32)); // out block 2 (o 8-11)
            acc3 = vdotq_s32(acc3, b, vld1q_s8(wt + ib * 64 + 48)); // out block 3 (o 12-15)
        }
        vst1q_s32(fc0_raw,      acc0);
        vst1q_s32(fc0_raw + 4,  acc1);
        vst1q_s32(fc0_raw + 8,  acc2);
        vst1q_s32(fc0_raw + 12, acc3);
#else
        // Scalar fallback (same vdotq weight layout, computed element-wise).
        for (int o = 0; o < NNUE_L0_SIZE; o++) fc0_raw[o] = bias[o];
        for (int ib = 0; ib < NNUE_L0_INPUT / 4; ib++) {
            for (int j = 0; j < 4; j++) {
                int32_t v = (int32_t)l0_in[ib * 4 + j];
                for (int ob = 0; ob < NNUE_L0_SIZE / 4; ob++)
                    for (int k = 0; k < 4; k++)
                        fc0_raw[ob * 4 + k] += v * (int32_t)wt[ib * 64 + ob * 16 + k * 4 + j];
            }
        }
#endif
    }

    // 3. Dual activation of FC0 outputs 0..L0_DIRECT-1 → 2×L0_DIRECT values for FC1
    //    fc1_in[0..14]  = SqrCReLU(fc0_raw[0..14])
    //    fc1_in[15..29] = CReLU(fc0_raw[0..14])
    //    fc1_in[30..31] = 0 (padding)
    //
    //    Stockfish SqrClippedReLU (sqr_clipped_relu.h):
    //      output = clamp(0, 127, (input * input) >> (2*WeightScaleBits + SqrShift))
    //             = clamp(0, 127, fc0^2 >> 19)
    //    Squaring the raw int32 value before clamping is essential — negative inputs
    //    produce non-zero squared outputs.  Clamping BEFORE squaring (old code) was
    //    wrong because it zeroed all negative fc0 values.
    int8_t fc1_in[NNUE_L1_PADDED];
    memset(fc1_in, 0, sizeof(fc1_in));
    for (int o = 0; o < NNUE_L0_DIRECT; o++) {
        // SqrCReLU: square raw value (works for negative), shift by 2*6+7=19, clamp
        int64_t sq64 = (int64_t)fc0_raw[o] * fc0_raw[o] >>
                       (2 * NNUE_WEIGHT_SHIFT + NNUE_SQR_SHIFT);
        fc1_in[o] = (int8_t)(sq64 > 127 ? 127 : sq64);
        // CReLU: shift and clamp to [0, 127]
        int v = fc0_raw[o] >> NNUE_WEIGHT_SHIFT;
        fc1_in[NNUE_L0_DIRECT + o] = (int8_t)(v < 0 ? 0 : v > 127 ? 127 : v);
    }

    // 4. FC1: 30(→32 padded) → 32
    int8_t fc2_in[NNUE_L1_SIZE];
    {
        const int32_t *bias = l1_biases[stack];
        const int8_t  *wt   = l1_weights[stack];
#ifdef __ARM_FEATURE_DOTPROD
        // vdotq layout: wt[ib*128 + ob*16 + k*4 + j] (8 input blocks × 8 output blocks)
        int32x4_t a0 = vld1q_s32(bias),      a1 = vld1q_s32(bias + 4);
        int32x4_t a2 = vld1q_s32(bias + 8),  a3 = vld1q_s32(bias + 12);
        int32x4_t a4 = vld1q_s32(bias + 16), a5 = vld1q_s32(bias + 20);
        int32x4_t a6 = vld1q_s32(bias + 24), a7 = vld1q_s32(bias + 28);
        for (int ib = 0; ib < NNUE_L1_PADDED / 4; ib++) {  // 8 iterations
            int8x16_t b = vreinterpretq_s8_s32(vdupq_n_s32(*(const int32_t *)(fc1_in + ib * 4)));
            a0 = vdotq_s32(a0, b, vld1q_s8(wt + ib * 128));
            a1 = vdotq_s32(a1, b, vld1q_s8(wt + ib * 128 + 16));
            a2 = vdotq_s32(a2, b, vld1q_s8(wt + ib * 128 + 32));
            a3 = vdotq_s32(a3, b, vld1q_s8(wt + ib * 128 + 48));
            a4 = vdotq_s32(a4, b, vld1q_s8(wt + ib * 128 + 64));
            a5 = vdotq_s32(a5, b, vld1q_s8(wt + ib * 128 + 80));
            a6 = vdotq_s32(a6, b, vld1q_s8(wt + ib * 128 + 96));
            a7 = vdotq_s32(a7, b, vld1q_s8(wt + ib * 128 + 112));
        }
        // Shift by WEIGHT_SHIFT=6, clamp to [0,127], narrow int32→int16→int8
        const int32x4_t zero32   = vdupq_n_s32(0);
        const int32x4_t max127_32 = vdupq_n_s32(127);
#define SHR6CLAMP(v) vminq_s32(vmaxq_s32(vshrq_n_s32(v, 6), zero32), max127_32)
        int32x4_t c0=SHR6CLAMP(a0), c1=SHR6CLAMP(a1), c2=SHR6CLAMP(a2), c3=SHR6CLAMP(a3);
        int32x4_t c4=SHR6CLAMP(a4), c5=SHR6CLAMP(a5), c6=SHR6CLAMP(a6), c7=SHR6CLAMP(a7);
#undef SHR6CLAMP
        int8x8_t b01 = vmovn_s16(vcombine_s16(vmovn_s32(c0), vmovn_s32(c1)));
        int8x8_t b23 = vmovn_s16(vcombine_s16(vmovn_s32(c2), vmovn_s32(c3)));
        int8x8_t b45 = vmovn_s16(vcombine_s16(vmovn_s32(c4), vmovn_s32(c5)));
        int8x8_t b67 = vmovn_s16(vcombine_s16(vmovn_s32(c6), vmovn_s32(c7)));
        vst1q_s8(fc2_in,      vcombine_s8(b01, b23));
        vst1q_s8(fc2_in + 16, vcombine_s8(b45, b67));
#else
        for (int o = 0; o < NNUE_L1_SIZE; o++) {
            int32_t sum = bias[o];
            const int8_t *row = wt + o * NNUE_L1_PADDED;
            for (int i = 0; i < NNUE_L1_PADDED; i++)
                sum += (int32_t)fc1_in[i] * (int32_t)row[i];
            sum >>= NNUE_WEIGHT_SHIFT;
            fc2_in[o] = (int8_t)(sum < 0 ? 0 : sum > 127 ? 127 : sum);
        }
#endif
    }

    // 5. FC2 (output): 32 → 1
    int32_t fc2_out = out_biases[stack];
    for (int i = 0; i < NNUE_L2_PADDED; i++)
        fc2_out += (int32_t)fc2_in[i] * (int32_t)out_weights[stack][i];

    // 6. Passthrough: FC0 output 15 is scaled and added to FC2 result.
    //    Stockfish formula: fwdOut = fc0_raw[15] * (600*OutputScale) / (127*WeightScaleBits)
    //                              = fc0_raw[15] * 9600 / 8128
    int32_t fwdOut = (int32_t)((int64_t)fc0_raw[NNUE_L0_DIRECT] * 9600 / 8128);
    int32_t positional = fc2_out + fwdOut;

    // 7. PSQT + final score
    //    Stockfish: value = (psqt_diff/2 + positional) / OutputScale(16)  [internal Value units]
    //    To convert to centipawns: * 100 / NormalizeToPawnValue(361)
    //    Combined: * 100 / (16 * 361) = * 100 / 5776
    int32_t psqt_diff = acc.psqt[stm][stack] - acc.psqt[stm ^ 1][stack];
#ifdef NNUE_PSQT_ONLY
    int score = (int32_t)((int64_t)(psqt_diff / 2) * 100 / 5776);
#else
    int score = (int32_t)((int64_t)(psqt_diff / 2 + positional) * 100 / 5776);
#endif

    if (getenv("NNUE_DEBUG"))
        fprintf(stderr, "NNUE_DEBUG: stack=%d fc2=%d fwd=%d positional=%d psqt_diff=%d total=%d\n",
                stack, fc2_out, fwdOut, positional, psqt_diff, score);

    if (getenv("NNUE_DEBUG_VERBOSE")) {
        fprintf(stderr, "NNUE_VERBOSE: stack=%d stm=%d pc=%d\n", stack, stm, piece_count);
        // Raw accumulator (first 16 of each perspective)
        fprintf(stderr, "  acc[stm][0..15]:");
        for (int i = 0; i < 16; i++) fprintf(stderr, " %d", (int)acc.acc[stm][i]);
        fprintf(stderr, "\n  acc[opp][0..15]:");
        for (int i = 0; i < 16; i++) fprintf(stderr, " %d", (int)acc.acc[stm^1][i]);
        fprintf(stderr, "\n");
        // FT activation output — print all non-zero values
        fprintf(stderr, "  l0_in nonzero:");
        for (int i = 0; i < NNUE_L0_INPUT; i++)
            if (l0_in[i]) fprintf(stderr, " [%d]=%d", i, (int)l0_in[i]);
        fprintf(stderr, "\n");
        // FC0 raw outputs
        fprintf(stderr, "  fc0_raw[0..15]:");
        for (int i = 0; i < NNUE_L0_SIZE; i++) fprintf(stderr, " %d", fc0_raw[i]);
        fprintf(stderr, "\n");
        // FC1 inputs (all 30 active + 2 padding)
        fprintf(stderr, "  fc1_in[0..31] (sqr|clip|pad):");
        for (int i = 0; i < NNUE_L1_PADDED; i++) fprintf(stderr, " %d", (int)fc1_in[i]);
        fprintf(stderr, "\n");
        // FC2 inputs (FC1 CReLU outputs)
        fprintf(stderr, "  fc2_in[0..31]:");
        for (int i = 0; i < NNUE_L1_SIZE; i++) fprintf(stderr, " %d", (int)fc2_in[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "  fc2_out=%d fwdOut=%d positional=%d psqt_diff=%d score=%d\n",
                fc2_out, fwdOut, positional, psqt_diff, score);
    }
    return score;
}

// ===========================================================================
// TDLeaf(λ) support — compiled only when TDLEAF=1
// ===========================================================================
#if TDLEAF

#include <cmath>
#include "tdleaf.h"
#include "hash.h"   // score_table, SCORE_SIZE, score_rec — for cache invalidation

// ---------------------------------------------------------------------------
// FP32 shadow copies of FC weights (natural output-major layout)
// ---------------------------------------------------------------------------
static float l0_weights_f32[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT];
static float l0_biases_f32 [NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static float l1_weights_f32[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED];
static float l1_biases_f32 [NNUE_LAYER_STACKS][NNUE_L1_SIZE];
static float l2_weights_f32[NNUE_LAYER_STACKS][NNUE_L2_PADDED];
static float l2_bias_f32   [NNUE_LAYER_STACKS];

// Gradient accumulators (zeroed before each game, filled by nnue_accumulate_gradients)
static float grad_l0_w[NNUE_LAYER_STACKS][NNUE_L0_SIZE * NNUE_L0_INPUT];
static float grad_l0_b[NNUE_LAYER_STACKS][NNUE_L0_SIZE];
static float grad_l1_w[NNUE_LAYER_STACKS][NNUE_L1_SIZE * NNUE_L1_PADDED];
static float grad_l1_b[NNUE_LAYER_STACKS][NNUE_L1_SIZE];
static float grad_l2_w[NNUE_LAYER_STACKS][NNUE_L2_PADDED];
static float grad_l2_b[NNUE_LAYER_STACKS];

// ---------------------------------------------------------------------------
// nnue_init_fp32_weights — dequantize int8 → float after nnue_load()
// ---------------------------------------------------------------------------
void nnue_init_fp32_weights()
{
    // Store weights at raw int8/int32 scale (no q_scale division).
    // The float forward pass uses the same arithmetic as the int path:
    //   fc_raw = bias_int32 + sum(w_int8 * input_[0,127])
    //   activation = clamp(fc_raw / 64, 0, 127)   (CReLU / SqrCReLU)
    // Requantisation is then just round(clamp(w_f32, -127, 127)).
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        // FC0: vdotq layout → natural [o * L0_INPUT + i]
        for (int o = 0; o < NNUE_L0_SIZE; o++) {
            int ob = o / 4, k = o % 4;
            l0_biases_f32[s][o] = (float)l0_biases[s][o];
            for (int i = 0; i < NNUE_L0_INPUT; i++) {
                int ib = i / 4, j = i % 4;
                l0_weights_f32[s][o * NNUE_L0_INPUT + i] =
                    (float)l0_weights[s][ib * 64 + ob * 16 + k * 4 + j];
            }
        }
        // FC1: vdotq layout → natural [o * PADDED + i]
        for (int o = 0; o < NNUE_L1_SIZE; o++) {
            int ob = o / 4, k = o % 4;
            l1_biases_f32[s][o] = (float)l1_biases[s][o];
            for (int i = 0; i < NNUE_L1_PADDED; i++) {
                int ib = i / 4, j = i % 4;
                l1_weights_f32[s][o * NNUE_L1_PADDED + i] =
                    (float)l1_weights[s][ib * 128 + ob * 16 + k * 4 + j];
            }
        }
        // FC2: natural layout (32 weights, 1 output)
        l2_bias_f32[s] = (float)out_biases[s];
        for (int i = 0; i < NNUE_L2_PADDED; i++)
            l2_weights_f32[s][i] = (float)out_weights[s][i];
    }
    memset(grad_l0_w, 0, sizeof(grad_l0_w));
    memset(grad_l0_b, 0, sizeof(grad_l0_b));
    memset(grad_l1_w, 0, sizeof(grad_l1_w));
    memset(grad_l1_b, 0, sizeof(grad_l1_b));
    memset(grad_l2_w, 0, sizeof(grad_l2_w));
    memset(grad_l2_b, 0, sizeof(grad_l2_b));
    printf("NNUE TDLeaf: FP32 weights initialised (%d stacks)\n", NNUE_LAYER_STACKS);
}

// ---------------------------------------------------------------------------
// nnue_forward_fp32 — FP32 forward pass, saves activations for backprop
// ---------------------------------------------------------------------------
void nnue_forward_fp32(const int16_t acc[2][NNUE_HALF_DIMS],
                       const int32_t psqt[2][NNUE_PSQT_BKTS],
                       bool wtm, NNUEActivations &act)
{
    int stm = wtm ? 1 : 0;  // WHITE=1, BLACK=0

    // 1. SqrCReLU from raw accumulator: pair (acc[i], acc[i+512]) for each perspective
    //    Layout: [stm_512 | opp_512]
    for (int p = 0; p < 2; p++) {
        int persp = (p == 0) ? stm : (stm ^ 1);
        float *out = act.l0_in + p * 512;
        const int16_t *a = acc[persp];
        for (int i = 0; i < 512; i++) {
            float va = (float)a[i];
            float vb = (float)a[i + 512];
            if (va < 0.0f) va = 0.0f; else if (va > 127.0f) va = 127.0f;
            if (vb < 0.0f) vb = 0.0f; else if (vb > 127.0f) vb = 127.0f;
            float sq = (va * vb) * (1.0f / 128.0f);  // >> 7
            out[i] = (sq > 127.0f) ? 127.0f : sq;
        }
    }

    // Layer stack from piece count is not available here; use stored stack index.
    // The caller fills act.stack before calling this function.
    int s = act.stack;

    // 2. FC0: [L0_INPUT → L0_SIZE] with FP32 weights
    for (int o = 0; o < NNUE_L0_SIZE; o++) {
        float sum = l0_biases_f32[s][o];
        const float *w = &l0_weights_f32[s][o * NNUE_L0_INPUT];
        for (int i = 0; i < NNUE_L0_INPUT; i++)
            sum += w[i] * act.l0_in[i];
        act.fc0_raw[o] = sum;
    }

    // 3. Dual activation of FC0 outputs 0..14 → FC1 input [0..29], padding [30..31]=0
    //    SqrCReLU: clamp(0,127, fc0_raw^2 >> 19)   (same formula as int path)
    //    CReLU:    clamp(0,127, fc0_raw >> 6)
    for (int o = 0; o < NNUE_L0_DIRECT; o++) {
        float raw = act.fc0_raw[o];
        // SqrCReLU (using float arithmetic, equivalent to int version)
        float sq = (raw * raw) * (1.0f / (1 << 19));
        act.fc1_in[o] = (sq < 0.0f) ? 0.0f : (sq > 127.0f) ? 127.0f : sq;
        // CReLU
        float v = raw * (1.0f / 64.0f);
        act.fc1_in[NNUE_L0_DIRECT + o] = (v < 0.0f) ? 0.0f : (v > 127.0f) ? 127.0f : v;
    }
    act.fc1_in[NNUE_L0_DIRECT * 2]     = 0.0f;
    act.fc1_in[NNUE_L0_DIRECT * 2 + 1] = 0.0f;

    // 4. FC1: [L1_PADDED → L1_SIZE]
    for (int o = 0; o < NNUE_L1_SIZE; o++) {
        float sum = l1_biases_f32[s][o];
        const float *w = &l1_weights_f32[s][o * NNUE_L1_PADDED];
        for (int i = 0; i < NNUE_L1_PADDED; i++)
            sum += w[i] * act.fc1_in[i];
        act.fc1_raw[o] = sum;
    }

    // 5. CReLU on FC1 output → FC2 input
    for (int o = 0; o < NNUE_L1_SIZE; o++) {
        float v = act.fc1_raw[o] * (1.0f / 64.0f);
        act.fc2_in[o] = (v < 0.0f) ? 0.0f : (v > 127.0f) ? 127.0f : v;
    }

    // 6. FC2: [L2_PADDED → 1]
    float fc2 = l2_bias_f32[s];
    for (int i = 0; i < NNUE_L2_PADDED; i++)
        fc2 += l2_weights_f32[s][i] * act.fc2_in[i];
    act.fc2_raw = fc2;

    // 7. Passthrough + final positional
    act.fwdOut    = act.fc0_raw[NNUE_L0_DIRECT] * (9600.0f / 8128.0f);
    act.positional = act.fc2_raw + act.fwdOut;

    // PSQT diff is not included in activations (it is constant w.r.t. FC weights)
    (void)psqt;
}

// ---------------------------------------------------------------------------
// nnue_accumulate_gradients — backprop one position, add to grad arrays
// grad_scale = alpha * e_t * d_t * (1-d_t) / K * (100 / 5776)
// ---------------------------------------------------------------------------
void nnue_accumulate_gradients(const NNUEActivations &act, float grad_scale)
{
    int s = act.stack;

    // 7. ∂loss/∂positional = grad_scale  (d_score_cp/d_positional = 100/5776 absorbed)
    float g_pos = grad_scale;

    // 6. FC2 backward: g_pos → grad of fc2_raw = g_pos
    float g_fc2_raw = g_pos;
    // grad w.r.t. FC2 weights and bias
    for (int i = 0; i < NNUE_L2_PADDED; i++)
        grad_l2_w[s][i] += g_fc2_raw * act.fc2_in[i];
    grad_l2_b[s] += g_fc2_raw;
    // grad w.r.t. fc2_in[i]
    float g_fc2_in[NNUE_L2_PADDED];
    for (int i = 0; i < NNUE_L2_PADDED; i++)
        g_fc2_in[i] = g_fc2_raw * l2_weights_f32[s][i];

    // 5. CReLU inverse: ∂fc2_in/∂fc1_raw = 1/64 if fc1_raw in (0, 127*64)
    float g_fc1_raw[NNUE_L1_SIZE];
    for (int o = 0; o < NNUE_L1_SIZE; o++) {
        float v = act.fc1_raw[o] * (1.0f / 64.0f);
        g_fc1_raw[o] = (v > 0.0f && v < 127.0f) ? g_fc2_in[o] * (1.0f / 64.0f) : 0.0f;
    }

    // 4. FC1 backward
    for (int o = 0; o < NNUE_L1_SIZE; o++) {
        if (g_fc1_raw[o] == 0.0f) continue;
        for (int i = 0; i < NNUE_L1_PADDED; i++)
            grad_l1_w[s][o * NNUE_L1_PADDED + i] += g_fc1_raw[o] * act.fc1_in[i];
        grad_l1_b[s][o] += g_fc1_raw[o];
    }
    // grad w.r.t. fc1_in[i]
    float g_fc1_in[NNUE_L1_PADDED] = {};
    for (int o = 0; o < NNUE_L1_SIZE; o++) {
        if (g_fc1_raw[o] == 0.0f) continue;
        const float *w = &l1_weights_f32[s][o * NNUE_L1_PADDED];
        for (int i = 0; i < NNUE_L1_PADDED; i++)
            g_fc1_in[i] += g_fc1_raw[o] * w[i];
    }

    // 3. Dual-activation inverse → g_fc0_raw[0..14]
    float g_fc0_raw[NNUE_L0_SIZE] = {};
    for (int o = 0; o < NNUE_L0_DIRECT; o++) {
        float raw = act.fc0_raw[o];
        // SqrCReLU gradient: d(clamp(0,127, raw^2/2^19))/d(raw) = 2*raw/2^19 when in range
        float sq = (raw * raw) * (1.0f / (1 << 19));
        if (sq > 0.0f && sq < 127.0f)
            g_fc0_raw[o] += g_fc1_in[o] * 2.0f * raw * (1.0f / (float)(1 << 19));
        // CReLU gradient: d(clamp(0,127, raw/64))/d(raw) = 1/64 when in range
        float v = raw * (1.0f / 64.0f);
        if (v > 0.0f && v < 127.0f)
            g_fc0_raw[o] += g_fc1_in[NNUE_L0_DIRECT + o] * (1.0f / 64.0f);
    }
    // Passthrough output[15]: ∂fwdOut/∂fc0_raw[15] = 9600/8128
    g_fc0_raw[NNUE_L0_DIRECT] += g_pos * (9600.0f / 8128.0f);

    // 2. FC0 backward (stop here — not training FT)
    for (int o = 0; o < NNUE_L0_SIZE; o++) {
        if (g_fc0_raw[o] == 0.0f) continue;
        for (int i = 0; i < NNUE_L0_INPUT; i++)
            grad_l0_w[s][o * NNUE_L0_INPUT + i] += g_fc0_raw[o] * act.l0_in[i];
        grad_l0_b[s][o] += g_fc0_raw[o];
    }
}

// ---------------------------------------------------------------------------
// nnue_apply_gradients — update FP32 weights from accumulators, then zero them
// ---------------------------------------------------------------------------
void nnue_apply_gradients()
{
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        for (int i = 0; i < NNUE_L0_SIZE * NNUE_L0_INPUT; i++)
            l0_weights_f32[s][i] -= grad_l0_w[s][i];
        for (int i = 0; i < NNUE_L0_SIZE; i++)
            l0_biases_f32[s][i] -= grad_l0_b[s][i];
        for (int i = 0; i < NNUE_L1_SIZE * NNUE_L1_PADDED; i++)
            l1_weights_f32[s][i] -= grad_l1_w[s][i];
        for (int i = 0; i < NNUE_L1_SIZE; i++)
            l1_biases_f32[s][i] -= grad_l1_b[s][i];
        for (int i = 0; i < NNUE_L2_PADDED; i++)
            l2_weights_f32[s][i] -= grad_l2_w[s][i];
        l2_bias_f32[s] -= grad_l2_b[s];
    }
    memset(grad_l0_w, 0, sizeof(grad_l0_w));
    memset(grad_l0_b, 0, sizeof(grad_l0_b));
    memset(grad_l1_w, 0, sizeof(grad_l1_w));
    memset(grad_l1_b, 0, sizeof(grad_l1_b));
    memset(grad_l2_w, 0, sizeof(grad_l2_w));
    memset(grad_l2_b, 0, sizeof(grad_l2_b));
}

// ---------------------------------------------------------------------------
// nnue_requantize_fc — FP32 → int8 for the live inference arrays
// ---------------------------------------------------------------------------
void nnue_requantize_fc()
{
    // Weights are stored at raw int8/int32 scale, so requantise by simply
    // rounding — no multiplication by q_scale or b_scale needed.
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        // FC0 weights: natural → vdotq layout
        for (int o = 0; o < NNUE_L0_SIZE; o++) {
            int ob = o / 4, k = o % 4;
            int64_t bq = (int64_t)roundf(l0_biases_f32[s][o]);
            l0_biases[s][o] = (int32_t)((bq < -2147483647) ? -2147483647 :
                                         (bq >  2147483647) ?  2147483647 : bq);
            for (int i = 0; i < NNUE_L0_INPUT; i++) {
                int ib = i / 4, j = i % 4;
                int q = (int)roundf(l0_weights_f32[s][o * NNUE_L0_INPUT + i]);
                if (q < -127) q = -127; if (q > 127) q = 127;
                l0_weights[s][ib * 64 + ob * 16 + k * 4 + j] = (int8_t)q;
            }
        }
        // FC1 weights: natural → vdotq layout
        for (int o = 0; o < NNUE_L1_SIZE; o++) {
            int ob = o / 4, k = o % 4;
            int64_t bq = (int64_t)roundf(l1_biases_f32[s][o]);
            l1_biases[s][o] = (int32_t)((bq < -2147483647) ? -2147483647 :
                                          (bq >  2147483647) ?  2147483647 : bq);
            for (int i = 0; i < NNUE_L1_PADDED; i++) {
                int ib = i / 4, j = i % 4;
                int q = (int)roundf(l1_weights_f32[s][o * NNUE_L1_PADDED + i]);
                if (q < -127) q = -127; if (q > 127) q = 127;
                l1_weights[s][ib * 128 + ob * 16 + k * 4 + j] = (int8_t)q;
            }
        }
        // FC2 weights: natural layout (no reorder needed)
        int64_t bq = (int64_t)roundf(l2_bias_f32[s]);
        out_biases[s] = (int32_t)((bq < -2147483647) ? -2147483647 :
                                   (bq >  2147483647) ?  2147483647 : bq);
        for (int i = 0; i < NNUE_L2_PADDED; i++) {
            int q = (int)roundf(l2_weights_f32[s][i]);
            if (q < -127) q = -127; if (q > 127) q = 127;
            out_weights[s][i] = (int8_t)q;
        }
    }
    // Clear score hash — cached evaluations are now stale.
    if (score_table && SCORE_SIZE > 0)
        memset(score_table, 0, SCORE_SIZE * sizeof(score_rec));
}

// ---------------------------------------------------------------------------
// nnue_save_fc_weights / nnue_load_fc_weights — companion .tdleaf.bin file
// File layout: magic(4) + version(4) + 8 stacks × [FC0_b + FC0_w + FC1_b +
//              FC1_w + FC2_b + FC2_w]  all in int32/int8 as used by inference.
// ---------------------------------------------------------------------------
static const uint32_t TDLEAF_MAGIC   = 0x544D4C46u; // "TMLF"
static const uint32_t TDLEAF_VERSION = 1u;

bool nnue_save_fc_weights(const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "TDLeaf: cannot write %s\n", path); return false; }
    fwrite(&TDLEAF_MAGIC,   4, 1, f);
    fwrite(&TDLEAF_VERSION, 4, 1, f);
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        fwrite(l0_biases[s],  sizeof(int32_t), NNUE_L0_SIZE,                f);
        fwrite(l0_weights[s], sizeof(int8_t),  NNUE_L0_SIZE * NNUE_L0_INPUT, f);
        fwrite(l1_biases[s],  sizeof(int32_t), NNUE_L1_SIZE,                f);
        fwrite(l1_weights[s], sizeof(int8_t),  NNUE_L1_SIZE * NNUE_L1_PADDED, f);
        fwrite(&out_biases[s],  sizeof(int32_t), 1,              f);
        fwrite(out_weights[s],  sizeof(int8_t),  NNUE_L2_PADDED, f);
    }
    fclose(f);
    return true;
}

bool nnue_load_fc_weights(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic = 0, version = 0;
    fread(&magic,   4, 1, f);
    fread(&version, 4, 1, f);
    if (magic != TDLEAF_MAGIC || version != TDLEAF_VERSION) {
        fprintf(stderr, "TDLeaf: bad magic/version in %s\n", path);
        fclose(f); return false;
    }
    for (int s = 0; s < NNUE_LAYER_STACKS; s++) {
        if (fread(l0_biases[s],  sizeof(int32_t), NNUE_L0_SIZE, f) != (size_t)NNUE_L0_SIZE ||
            fread(l0_weights[s], sizeof(int8_t),  NNUE_L0_SIZE * NNUE_L0_INPUT, f)
                != (size_t)(NNUE_L0_SIZE * NNUE_L0_INPUT) ||
            fread(l1_biases[s],  sizeof(int32_t), NNUE_L1_SIZE, f) != (size_t)NNUE_L1_SIZE ||
            fread(l1_weights[s], sizeof(int8_t),  NNUE_L1_SIZE * NNUE_L1_PADDED, f)
                != (size_t)(NNUE_L1_SIZE * NNUE_L1_PADDED) ||
            fread(&out_biases[s],  sizeof(int32_t), 1, f) != 1 ||
            fread(out_weights[s],  sizeof(int8_t),  NNUE_L2_PADDED, f) != (size_t)NNUE_L2_PADDED) {
            fprintf(stderr, "TDLeaf: read error in %s (stack %d)\n", path, s);
            fclose(f); return false;
        }
    }
    fclose(f);
    // Sync FP32 copies from the newly loaded int8 arrays.
    nnue_init_fp32_weights();
    printf("TDLeaf: loaded FC weights from %s\n", path);
    return true;
}

#endif // TDLEAF
