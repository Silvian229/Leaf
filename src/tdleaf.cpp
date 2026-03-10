// Epoch TDLeaf(λ) online learning — implementation
// Compiled only when TDLEAF=1 (included by Epoch.cc after nnue.cpp).

#include "define.h"

#if TDLEAF

#include <cmath>
#include <cstring>
#include <cstdio>
#include "chess.h"
#include "nnue.h"
#include "tdleaf.h"

// ---------------------------------------------------------------------------
// tdleaf_record_ply — walk the PV to the leaf, then snapshot its accumulator
// ---------------------------------------------------------------------------
void tdleaf_record_ply(TDGameRecord &rec,
                       const position &root_pos,
                       const NNUEAccumulator &root_acc,
                       const move *pv,
                       int score_root_stm,
                       bool root_wtm,
                       const int *id_scores,
                       int id_score_count)
{
    if (rec.n_plies >= MAX_GAME_PLY) return;  // safety guard

    // Walk the PV, updating the position and accumulator incrementally.
    // We use two alternating accumulator slots to avoid unnecessary copies.
    NNUEAccumulator acc_a = root_acc;   // current leaf accumulator
    NNUEAccumulator acc_b;              // scratch for next step
    position cur = root_pos;
    int pv_len = 0;

    for (int k = 0; k < MAXD && pv[k].t != NOMOVE; k++) {
        position next = cur;
        if (!next.exec_move(pv[k], 0)) break;  // illegal — stop here
        nnue_record_delta(acc_b, cur, next, pv[k]);
        nnue_apply_delta(acc_b, acc_a, next);
        cur   = next;
        acc_a = acc_b;
        pv_len++;
    }
    // acc_a now holds the fully computed leaf accumulator; cur is the leaf position.

    bool leaf_wtm = root_wtm ^ (bool)(pv_len & 1);

    // Leaf piece count for stack selection.
    int pc = 2;  // kings
    for (int sd = 0; sd < 2; sd++)
        for (int pt = PAWN; pt <= QUEEN; pt++)
            pc += cur.plist[sd][pt][0];
    pc = (pc < 1) ? 1 : (pc > 32) ? 32 : pc;

    // Use the NNUE static evaluation of the leaf position directly.
    // This ensures d[t] is computed from what nnue_forward_fp32 actually produces
    // at that position, making the gradient self-consistent.
    // (The propagated search score includes quiescence and may differ.)
    int leaf_score_stm = nnue_evaluate(acc_a, (int)leaf_wtm, pc);

#if TDLEAF_CHECK_SCORE
    {
        // Sanity check: propagated root score (with per-ply sign flip) vs direct eval.
        int propagated = (pv_len & 1) ? -score_root_stm : score_root_stm;
        int diff = leaf_score_stm - propagated;
        fprintf(stderr, "TDLeaf check: pv_len=%d  leaf_wtm=%d  direct=%d  propagated=%d  diff=%d%s\n",
                pv_len, (int)leaf_wtm, leaf_score_stm, propagated, diff,
                (diff < -300 || diff > 300) ? "  *** LARGE ***" : "");
    }
#endif

    // Compute variance of the last N iterative-deepening scores.
    float id_var = 0.0f;
    if (id_score_count >= 2) {
        float id_mean = 0.0f;
        for (int i = 0; i < id_score_count; i++) id_mean += id_scores[i];
        id_mean /= id_score_count;
        for (int i = 0; i < id_score_count; i++) {
            float delta = id_scores[i] - id_mean;
            id_var += delta * delta;
        }
        id_var /= id_score_count;
    }

    TDRecord &r = rec.plies[rec.n_plies++];
    memcpy(r.acc[0],  acc_a.acc[0],  NNUE_HALF_DIMS  * sizeof(int16_t));
    memcpy(r.acc[1],  acc_a.acc[1],  NNUE_HALF_DIMS  * sizeof(int16_t));
    memcpy(r.psqt[0], acc_a.psqt[0], NNUE_PSQT_BKTS * sizeof(int32_t));
    memcpy(r.psqt[1], acc_a.psqt[1], NNUE_PSQT_BKTS * sizeof(int32_t));
    r.score_stm        = leaf_score_stm;
    r.wtm              = leaf_wtm;
    r.stack            = (pc - 1) / 4;
    r.id_score_variance = id_var;

    // Enumerate active features at the leaf position for FT/PSQT backprop.
    // Indices are by actual perspective (0=BLACK, 1=WHITE) matching halfkav2_feature().
    for (int p = 0; p < 2; p++) {
        int ksq = cur.plist[p][KING][1];
        r.n_ft[p] = 0;
        for (int sd = 0; sd < 2; sd++)
            for (int pt = PAWN; pt <= KING; pt++)
                for (int i = 1; i <= cur.plist[sd][pt][0]; i++) {
                    if (r.n_ft[p] >= NNUE_MAX_FT_PER_PERSP) goto ft_done;
                    int fi = halfkav2_feature(p, ksq, cur.plist[sd][pt][i], pt, sd);
                    if (fi >= 0) r.ft_idx[p][r.n_ft[p]++] = fi;
                }
        ft_done:;
    }
}

// ---------------------------------------------------------------------------
// tdleaf_update_after_game — run TDLeaf(λ) update for a completed game
// ---------------------------------------------------------------------------
void tdleaf_update_after_game(TDGameRecord &rec, float result, const char *save_path)
{
    int T = rec.n_plies;
    if (T < TDLEAF_MIN_PLIES) {
        fprintf(stderr, "TDLeaf: skipping short game (%d plies)\n", T);
        return;
    }

    // -----------------------------------------------------------------------
    // 1. Convert scores to White-POV sigmoid values d[t] ∈ (0,1)
    // -----------------------------------------------------------------------
    static float d[MAX_GAME_PLY];
    static float score_w_cp[MAX_GAME_PLY];
    for (int t = 0; t < T; t++) {
        score_w_cp[t] = rec.plies[t].wtm
                        ?  (float)rec.plies[t].score_stm
                        : -(float)rec.plies[t].score_stm;
        d[t] = 1.0f / (1.0f + expf(-score_w_cp[t] / TDLEAF_K));
    }

    // -----------------------------------------------------------------------
    // 2. Compute TD errors backward
    //    e[T-1] = result - d[T-1]
    //    e[t]   = (d[t+1] - d[t]) + lambda * e[t+1]
    // -----------------------------------------------------------------------
    static float e[MAX_GAME_PLY];
    e[T - 1] = result - d[T - 1];
    for (int t = T - 2; t >= 0; t--) {
        float delta_d  = d[t + 1] - d[t];
        float delta_cp = fabsf(score_w_cp[t + 1] - score_w_cp[t]);
        if (delta_cp > TDLEAF_SCORE_CLIP_CP && delta_cp > 0.0f)
            delta_d *= TDLEAF_SCORE_CLIP_CP / delta_cp;
        e[t] = delta_d + TDLEAF_LAMBDA * e[t + 1];
    }

    // -----------------------------------------------------------------------
    // 3. For each ply, run FP32 forward pass + accumulate gradients
    //    grad_scale = alpha * e[t] * d[t] * (1-d[t]) / K * (100/5776)
    //    (The 100/5776 converts the score output unit to centipawns for the
    //     sigmoid: sigmoid takes score_cp/K, so ∂sigmoid/∂weight includes
    //     the cp-to-raw conversion factor.)
    // -----------------------------------------------------------------------
    const float cp_factor = 100.0f / 5776.0f;  // ∂score_cp / ∂positional_raw

    for (int t = 0; t < T; t++) {
        float sig_grad = d[t] * (1.0f - d[t]) / TDLEAF_K;
        // positional is from STM perspective; score_white = (wtm ? +1 : -1) * positional * scale.
        // nnue_apply_gradients does w -= grad, so negate for wtm to get w += gradient.
        float wtm_sign = rec.plies[t].wtm ? -1.0f : 1.0f;
        // ID stability weight: down-weight positions whose score fluctuated across ID depths.
        float id_weight = 1.0f / (1.0f + rec.plies[t].id_score_variance / TDLEAF_ID_VAR_SIGMA2);
        float grad_scale = TDLEAF_ALPHA * e[t] * sig_grad * cp_factor * wtm_sign * id_weight;

        if (grad_scale == 0.0f) continue;

        NNUEActivations act;
        act.stack = rec.plies[t].stack;
        nnue_forward_fp32(rec.plies[t].acc, rec.plies[t].psqt,
                          rec.plies[t].wtm, act);
        // Fill FT/PSQT backprop fields before calling accumulate_gradients.
        memcpy(act.acc_raw[0], rec.plies[t].acc[0], NNUE_HALF_DIMS * sizeof(int16_t));
        memcpy(act.acc_raw[1], rec.plies[t].acc[1], NNUE_HALF_DIMS * sizeof(int16_t));
        act.n_ft[0] = rec.plies[t].n_ft[0];
        act.n_ft[1] = rec.plies[t].n_ft[1];
        memcpy(act.ft_idx[0], rec.plies[t].ft_idx[0], rec.plies[t].n_ft[0] * sizeof(int));
        memcpy(act.ft_idx[1], rec.plies[t].ft_idx[1], rec.plies[t].n_ft[1] * sizeof(int));
        // stm_persp is set inside nnue_forward_fp32 above.
        nnue_accumulate_gradients(act, grad_scale);
    }

    // -----------------------------------------------------------------------
    // 4. Apply gradients, requantize, and save
    // -----------------------------------------------------------------------
    nnue_apply_gradients();
    nnue_requantize_fc();

    if (save_path && save_path[0]) {
        if (!nnue_save_fc_weights(save_path))
            fprintf(stderr, "TDLeaf: failed to save weights to %s\n", save_path);
    }

    fprintf(stderr, "TDLeaf: updated weights for %d-ply game (result=%.1f)\n",
            T, (double)result);
}

#endif // TDLEAF
