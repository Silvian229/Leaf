// EXchess NNUE evaluation — HalfKAv2_hm format (Stockfish 15/16 era)
// Written from scratch; Stockfish source consulted only for file-format layout.
//
// File format empirically determined from nn-ae6a388e4a1a.nnue:
//   [Header] version(4) + hash(4) + desc_size(4) + desc(N)
//   [FT]     ft_hash(4) + LEB128(biases 1024 i16) + LEB128(weights 22528×1024 i16)
//             + LEB128(psqt 22528×8 i32)
//   [Stacks] 8 × [stack_hash(4) + FC0_bias(16×i32) + FC0_wt(16×3072×i8)
//                               + FC1_bias(32×i32) + FC1_wt(32×32×i8)
//                               + FC2_bias(1×i32)  + FC2_wt(32×i8)]
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

// FC0: output-major weight layout, 16 outputs × 3072 inputs
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
    // Own king selects the bucket only — it is NOT a feature.
    if (ptype == KING && pside == persp)
        return -1;

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
            //   For each 4-input block ib (0..767) and 4-output block ob (0..3),
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
// nnue_evaluate — forward pass → centipawns from stm's perspective
//
// Input preparation (dual activation — "paired CReLU"):
//   For each perspective (stm, opp), from NNUE_HALF_DIMS=1024 int16 accumulator:
//     - SqrCReLU on pairs (acc[i], acc[i+512]) for i=0..511:
//         a = clamp(acc[i]>>FT_SHIFT, 0, 127)
//         b = clamp(acc[i+512]>>FT_SHIFT, 0, 127)
//         sqr[i] = clamp((a*b)>>SQR_SHIFT, 0, 127)
//     - CReLU on all 1024: clip[i] = clamp(acc[i]>>FT_SHIFT, 0, 127)
//     - Per-side output: [sqr[0..511] | clip[0..1023]] = 1536 int8 values
//   Total FC0 input: [stm_1536 | opp_1536] = 3072 int8 values
//
// FC0 (16 outputs, output-major weights 16×3072):
//   raw[o] = bias[o] + dot(l0_in, weights[o*3072 .. (o+1)*3072-1])
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

    // 1. Dual activation: produce 3072 int8 values
    //    Layout: [stm_sqr(0..511) | stm_clip(0..1023) | opp_sqr(0..511) | opp_clip(0..1023)]
    int8_t l0_in[NNUE_L0_INPUT];
    const int16_t *persp_acc[2] = { acc.acc[stm], acc.acc[stm ^ 1] };

    // Fused dual-activation: SqrCReLU(a[0..511], a[512..1023]) → out[0..511]
    //                         CReLU(a[0..511])                   → out[512..1023]
    //                         CReLU(a[512..1023])                → out[1024..1535]
    // One pass reads each half of the accumulator exactly once per perspective.
#ifdef __ARM_NEON
    {
        const int16x8_t zero16   = vdupq_n_s16(0);
        const int16x8_t max127_16 = vdupq_n_s16(127);
        for (int p = 0; p < 2; p++) {
            const int16_t *a = persp_acc[p];
            int8_t *out = l0_in + p * 1536;
            for (int i = 0; i < 512; i += 8) {
                int16x8_t va16 = vshrq_n_s16(vld1q_s16(a + i),       6); // FT_SHIFT=6
                int16x8_t vb16 = vshrq_n_s16(vld1q_s16(a + i + 512), 6);
                int16x8_t va_c = vminq_s16(vmaxq_s16(va16, zero16), max127_16);
                int16x8_t vb_c = vminq_s16(vmaxq_s16(vb16, zero16), max127_16);
                int8x8_t  va8  = vmovn_s16(va_c);
                int8x8_t  vb8  = vmovn_s16(vb_c);
                // SqrCReLU: max product = 127*127=16129, >>7 = 126 ≤ 127, no clamp needed
                int8x8_t  sq8  = vmovn_s16(vshrq_n_s16(vmull_s8(va8, vb8), 7)); // SQR_SHIFT=7
                vst1_s8(out + i,        sq8); // SqrCReLU → [0..511]
                vst1_s8(out + 512 + i,  va8); // CReLU(a[0..511]) → [512..1023]
                vst1_s8(out + 1024 + i, vb8); // CReLU(a[512..1023]) → [1024..1535]
            }
        }
    }
#else
    for (int p = 0; p < 2; p++) {
        const int16_t *a = persp_acc[p];
        int8_t *out = l0_in + p * 1536;

        // SqrCReLU on pairs (i, i+512)
        for (int i = 0; i < 512; i++) {
            int va = (int)a[i]       >> NNUE_FT_SHIFT;
            int vb = (int)a[i + 512] >> NNUE_FT_SHIFT;
            if (va < 0) va = 0; else if (va > 127) va = 127;
            if (vb < 0) vb = 0; else if (vb > 127) vb = 127;
            int sq = (va * vb) >> NNUE_SQR_SHIFT;
            out[i] = (int8_t)(sq > 127 ? 127 : sq);
        }

        // CReLU on all 1024
        for (int i = 0; i < NNUE_HALF_DIMS; i++) {
            int v = (int)a[i] >> NNUE_FT_SHIFT;
            out[512 + i] = (int8_t)(v < 0 ? 0 : v > 127 ? 127 : v);
        }
    }
#endif

    // 2. FC0: 3072 → 16
    //    Weights are stored in vdotq-friendly layout (set at load time):
    //      wt[ib*64 + ob*16 + k*4 + j]  where ib=i/4, j=i%4, ob=o/4, k=o%4
    //    Each loop iteration processes 4 consecutive inputs against all 16 outputs
    //    via 4 vdotq_s32 calls — 768 iterations total (vs 16×768 per-output).
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
    int8_t fc1_in[NNUE_L1_PADDED];
    memset(fc1_in, 0, sizeof(fc1_in));
    for (int o = 0; o < NNUE_L0_DIRECT; o++) {
        int v = fc0_raw[o] >> NNUE_WEIGHT_SHIFT;
        if (v < 0) v = 0; else if (v > 127) v = 127;
        int sq = (v * v) >> NNUE_SQR_SHIFT;
        fc1_in[o]                  = (int8_t)(sq > 127 ? 127 : sq);  // SqrCReLU
        fc1_in[NNUE_L0_DIRECT + o] = (int8_t)v;                       // CReLU
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
    //    FC0 output 15 (passthrough) is right-shifted by WEIGHT_SHIFT (÷64) to match
    //    the scale of the FC2 dot product — same reduction as outputs 0-14 undergo
    //    before entering FC1.
    int32_t network_out = out_biases[stack];
    network_out += fc0_raw[NNUE_L0_DIRECT] >> NNUE_WEIGHT_SHIFT;  // passthrough ÷64
    for (int i = 0; i < NNUE_L2_PADDED; i++)
        network_out += (int32_t)fc2_in[i] * (int32_t)out_weights[stack][i];

    // 6. PSQT contribution for the selected stack
    int32_t psqt_out = acc.psqt[stm][stack] - acc.psqt[stm ^ 1][stack];

    int score = network_out / NNUE_CP_SCALE + psqt_out / NNUE_PSQT_SCALE;
    return score;
}
