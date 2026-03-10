// Epoch TDLeaf(λ) online learning for the NNUE FC layers.
//
// Algorithm (Baxter, Tridgell & Weaver, 2000):
//   After a game of T half-moves, let d_t = sigmoid(score_white_t / K).
//   TD errors (backward view):
//     e_{T-1} = result - d_{T-1}
//     e_t     = (d_{t+1} - d_t) + lambda * e_{t+1}
//   Weight update:
//     Δw = alpha * Σ_t  e_t * ∇_w d_t
//
// FC layers (FC0/FC1/FC2) and FT biases (1024 int16) are trained.  FT weights
// and PSQT are also trained (FT weights 46 MB, PSQT 720 KB).
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
static const float TDLEAF_LAMBDA          = 0.7f;   // eligibility trace decay
static const float TDLEAF_ALPHA           = 200.0f;  // learning rate for FC layers
// FT learning rate — much smaller than FC because FT weights are int16 (256× finer
// than int8) and are shared across all positions.
static const float TDLEAF_ALPHA_FT        = 200.0f; // learning rate for FT weights
// FT learning rate scale applied inside nnue_accumulate_gradients to g_acc[p][d].
static const float NNUE_FT_LR_SCALE = 1.000f;
// PSQT learning rate scale applied inside nnue_accumulate_gradients to g_psqt_diff.
// Separate from NNUE_FT_LR_SCALE so PSQT can be tuned independently.
// NOTE: grad_scale = TDLEAF_ALPHA × e[t] × d(1-d)/K × cp_factor.
//   With K=400 and cp_factor=100/5776, |grad_scale| ≈ 2e-4 for typical errors.
//   PSQT weights are at int32 scale (a pawn ≈ 5776 units), so a large multiplier
//   is needed to get meaningful per-game updates.
static const float NNUE_PSQT_LR_SCALE = 1000.0f;
// FT bias learning rate scale: applied to the FT bias gradient (g_acc[0][d] + g_acc[1][d]).
// FT biases are shared across all positions so per-game cancellation is significant,
// similar to FC biases. Start at 10× FT weight scale; tune empirically.
static const float NNUE_FT_BIAS_LR_SCALE = 10.0f;
// FC bias learning rate scale: applied to all FC bias gradients.
// FC biases are in int32 units (FC0: ~332, FC1: ~-400, FC2: ~1242).
// Unlike weights, bias gradients have no l0_in[i] multiplier, so TDLeaf's
// alternating wtm_sign × e[t] structure causes near-complete per-game cancellation.
// A large multiplier is needed to produce visible int32 updates.  Tune empirically.
static const float NNUE_FC_BIAS_LR_SCALE = 1000.0f;
//
static const float TDLEAF_K               = 400.0f; // sigmoid temperature (centipawns)
static const int   TDLEAF_MIN_PLIES       = 8;      // skip games shorter than this
// Per-update clamp: a single game's gradient update is limited to at most this
// fraction of the weight's current magnitude (min ref = 1.0 so near-zero weights
// can still move).  Set to 1.0 to disable.  Applied to both FC and FT layers.
static const float TDLEAF_MAX_UPDATE_FRAC = 1.0f; //0.10f;  // max 10% change per update
// Approach 1 — TD error clipping.
// When the white-POV score change between consecutive moves exceeds this
// threshold (centipawns), the (d[t+1]−d[t]) contribution to the eligibility
// trace is scaled down proportionally.  Set to a large value to disable.
static const float TDLEAF_SCORE_CLIP_CP  = 200.0f;
// Approach 2 — iterative-deepening score stability weight.
// w_t = 1 / (1 + id_score_variance / TDLEAF_ID_VAR_SIGMA2)
// Expressed in cp²: 10000 corresponds to a 100 cp std-dev reference.
// Larger values are more tolerant of ID score instability.
static const float TDLEAF_ID_VAR_SIGMA2  = 10000.0f;

// ---------------------------------------------------------------------------
// Per-ply record: accumulator snapshot + search score
// ---------------------------------------------------------------------------
struct TDRecord {
    int16_t acc [2][NNUE_HALF_DIMS];   // raw accumulator [perspective][dim]
    int32_t psqt[2][NNUE_PSQT_BKTS];  // PSQT sums [perspective][bucket]
    int     score_stm;                 // search score (centipawns, side-to-move POV)
    int     stack;                     // layer stack index used (piece_count-1)/4
    bool    wtm;                       // White to move at this position
    float   id_score_variance;         // variance of last N ID depth scores (cp²); 0 if < 2 depths
    // Active feature indices at the leaf position (indexed by actual perspective 0=BLACK,1=WHITE).
    // Used for FT and PSQT gradient backprop.
    int     ft_idx[2][NNUE_MAX_FT_PER_PERSP];
    int8_t  n_ft[2];
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
// Walks the principal variation to the leaf position; records the leaf
// accumulator, leaf wtm, and leaf-perspective score (not the root's).
//
// root_pos:        game.pos
// root_acc:        game.ts.tdata[0].n[0].acc
// pv:              game.ts.tdata[0].pc[0]   (NOMOVE-terminated)
// score_root_stm:  game.ts.g_last           (score, root STM perspective)
// root_wtm:        game.pos.wtm
void tdleaf_record_ply(TDGameRecord &rec,
                       const struct position &root_pos,
                       const NNUEAccumulator &root_acc,
                       const move *pv,
                       int score_root_stm,
                       bool root_wtm,
                       const int *id_scores,
                       int id_score_count);

// Run the full TDLeaf(λ) update after a game ends.
// result: game outcome from White's perspective (1.0=White wins, 0.5=draw, 0.0=Black wins).
// Calls nnue_apply_gradients(), nnue_requantize_fc(), and nnue_save_fc_weights().
void tdleaf_update_after_game(TDGameRecord &rec, float result, const char *save_path);

#endif // TDLEAF_H
