// EXchess TDLeaf(λ) online learning for the NNUE FC layers.
//
// Algorithm (Baxter, Tridgell & Weaver, 2000):
//   After a game of T half-moves, let d_t = sigmoid(score_white_t / K).
//   TD errors (backward view):
//     e_{T-1} = result - d_{T-1}
//     e_t     = (d_{t+1} - d_t) + lambda * e_{t+1}
//   Weight update:
//     Δw = alpha * Σ_t  e_t * ∇_w d_t
//
// Only the FC layers (FC0/FC1/FC2) are trained.  The FT (46 MB) is unchanged.
// FP32 shadow copies of the FC weights are maintained in nnue.cpp; after each
// game the int8 inference arrays are updated via nnue_requantize_fc().
//
// Build: perl comp.pl <version> NNUE=1 TDLEAF=1

#ifndef TDLEAF_H
#define TDLEAF_H

#include "define.h"
#include "nnue.h"

// ---------------------------------------------------------------------------
// Hyperparameters (can be overridden by setvalue/environment at runtime)
// ---------------------------------------------------------------------------
static const float TDLEAF_LAMBDA    = 0.7f;   // eligibility trace decay
static const float TDLEAF_ALPHA     = 200.0f; // learning rate (raw-int-scale weights)
static const float TDLEAF_K         = 400.0f; // sigmoid temperature (centipawns)
static const int   TDLEAF_MIN_PLIES = 8;      // skip games shorter than this

// ---------------------------------------------------------------------------
// Per-ply record: accumulator snapshot + search score
// ---------------------------------------------------------------------------
struct TDRecord {
    int16_t acc [2][NNUE_HALF_DIMS];   // raw accumulator [perspective][dim]
    int32_t psqt[2][NNUE_PSQT_BKTS];  // PSQT sums [perspective][bucket]
    int     score_stm;                 // search score (centipawns, side-to-move POV)
    int     stack;                     // layer stack index used (piece_count-1)/4
    bool    wtm;                       // White to move at this position
};

// ---------------------------------------------------------------------------
// Per-game record: array of TDRecord entries + outcome
// ---------------------------------------------------------------------------
struct TDGameRecord {
    TDRecord plies[MAX_GAME_PLY];
    int      n_plies;
    // n_plies is reset to 0 at game start; entries filled by tdleaf_record_ply()
};

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

// Record one ply after each ts.search() call.
// root_acc: root search node accumulator (game.ts.tdata[0].n[0].acc).
// score_stm: raw search score from STM POV (game.ts.g_last).
// wtm: game.pos.wtm (before making the move).
// piece_count: total pieces on board for stack selection.
void tdleaf_record_ply(TDGameRecord &rec,
                       const NNUEAccumulator &root_acc,
                       int score_stm, bool wtm, int piece_count);

// Run the full TDLeaf(λ) update after a game ends.
// result: game outcome from White's perspective (1.0=White wins, 0.5=draw, 0.0=Black wins).
// Calls nnue_apply_gradients(), nnue_requantize_fc(), and nnue_save_fc_weights().
void tdleaf_update_after_game(TDGameRecord &rec, float result, const char *save_path);

#endif // TDLEAF_H
