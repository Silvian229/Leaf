// EXchess NNUE evaluation — HalfKAv2_hm format (Stockfish 15/16 era)
//
// Architecture (determined empirically from file structure):
//   Features = HalfKAv2_hm — king-square bucket × all-piece-square
//   Feature space : 22,528  (32 king-buckets × 704 piece-sq indices)
//   Accumulator   : 1024 int16 per perspective  +  8 int32 PSQT per perspective
//   Network       : 8 layer-stacks (selected by material count), each:
//     - FC0: 3072 → 16  (3072 = 2 × 1536 dual-activation output per side)
//     - FC1: 30   → 32  (30 = 15 CReLU + 15 SqrCReLU from FC0 outputs 0-14)
//     - FC2: 32   → 1   (output; FC0 output-15 adds directly)
//   PSQT : accumulated separately; blended with network output
//
// Compatible net: nn-ae6a388e4a1a.nnue (official-stockfish/networks)

#ifndef NNUE_H
#define NNUE_H

#include <cstdint>

// ---------------------------------------------------------------------------
// Architecture constants
// ---------------------------------------------------------------------------
static const int NNUE_HALF_DIMS    = 1024;  // accumulator units per perspective
static const int NNUE_FT_INPUTS    = 22528; // 32 king-buckets × 704 piece-sq
static const int NNUE_LAYER_STACKS = 8;     // separate nets per material bucket
static const int NNUE_PSQT_BKTS   = 8;     // PSQT buckets (== LAYER_STACKS)

// FC0: sparse-input affine (dual-activation input: 1536 per side × 2 sides)
static const int NNUE_L0_SIZE     = 16;   // FC0 output neurons (incl. direct-out)
static const int NNUE_L0_DIRECT   = 15;   // FC0 outputs going through activations
static const int NNUE_L0_INPUT    = 3072; // = 2 × (512 sqr + 1024 clip) per side

// FC1: dense (takes dual-activation of FC0 outputs 0..14 → 15 sqr + 15 clip = 30)
static const int NNUE_L1_SIZE     = 32;   // FC1 output neurons
static const int NNUE_L1_PADDED   = 32;   // padded input dim (ceil(30, 16) = 32)

// FC2 (output): input = NNUE_L1_SIZE = 32 neurons
static const int NNUE_L2_PADDED   = 32;   // padded input dim for FC2

// Quantization scales
static const int NNUE_FT_SHIFT     = 6;   // FT: int16 >> 6 → [0,127] int8
static const int NNUE_WEIGHT_SHIFT = 6;   // FC weights: accumulator >> 6 → [0,127]
static const int NNUE_SQR_SHIFT    = 7;   // SqrCReLU: (v*v) >> 7 → [0,127]
// Scale to convert raw network output to EXchess internal units (pawn=100):
//   The FC0 passthrough (output 15) must be right-shifted by WEIGHT_SHIFT (÷64)
//   before adding to the FC2 output — same reduction applied to outputs 0-14.
//   After that shift, network_out / 128 gives EXchess centipawns.
//   PSQT: empirically calibrated — one pawn contributes ~6500 PSQT units;
//   dividing by 64 gives ~100 cp per pawn.  The "÷2 per perspective" that
//   Stockfish applies internally is already baked into the stored weight values.
static const int NNUE_CP_SCALE     = 128; // network_out / 128 → EXchess centipawns
static const int NNUE_PSQT_SCALE   =  64; // psqt_out / 64  → EXchess centipawns

// ---------------------------------------------------------------------------
// Accumulator (one per search node, copied from parent then updated)
// ---------------------------------------------------------------------------
struct NNUEAccumulator {
    int16_t acc [2][NNUE_HALF_DIMS];  // [perspective][unit]  WHITE=1, BLACK=0
    int32_t psqt[2][NNUE_PSQT_BKTS]; // [perspective][bucket]
    bool    dirty[2];                 // true → full refresh needed

    NNUEAccumulator() { dirty[0] = dirty[1] = true; }
};

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------
extern bool nnue_available;

// Load a HalfKAv2_hm .nnue file. Returns true on success.
bool nnue_load(const char *path);

// Full accumulator refresh from the current position.
void nnue_init_accumulator(NNUEAccumulator &acc, const struct position &pos);

// Incremental update after exec_move (copy-make search style).
void nnue_update_accumulator(NNUEAccumulator &next_acc,
                             const struct position &before,
                             const struct position &after,
                             union move mv);

// Forward pass. Returns centipawns from side-to-move's perspective.
// piece_count: total pieces on board (for layer-stack selection).
int nnue_evaluate(const NNUEAccumulator &acc, int stm, int piece_count);

#endif // NNUE_H
