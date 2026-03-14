// EXchess source code, (c) Daniel C. Homan  1997-2017
// Released under the GNU public license, see file license.txt

/* Search Functions */
//-----------------------------------------------------------------------------
// move tree_search::search(position p, int time_limit, int T, game_rec *gr)
// inline void tree_search::pc_update(move pcmove, int ply)
// inline void MSort(move_rec *Lb, move_rec *Ub)
// int search_node::root_pvs(int alpha, int beta, int depth)
// int search_node::pvs(int alpha, int beta, int depth)
// int search_node::qsearch(int alpha, int beta, int qply)
//-----------------------------------------------------------------------------

#include <iostream>
#include <cstdio>
#include <fstream>
#include <assert.h>
#include <pthread.h>

#if DEBUG_SEARCH
 #include <string.h>
 #include <stdlib.h>
#endif

#include "define.h"
#include "chess.h"
#include "const.h"
#include "funct.h"
#include "hash.h"
#include "search.h"
#include "extern.h"

// ---------------------------------------------------------------------------
// LMR precomputed lookup tables — eliminates repeated integer division inside
// the move loop.  Indexed by depth (0..MAXD) and move index (0..MAX_MOVES-1).
// Initialised once at startup by lmr_init_tables() called from open_hash().
// ---------------------------------------------------------------------------
static int lmr_d5 [MAXD + 1];     // depth / 5
static int lmr_d8 [MAXD + 1];     // depth / 8
static int lmr_d12[MAXD + 1];     // depth / 12
static int lmr_mi8[MAX_MOVES];    // move_index / 8

void lmr_init_tables()
{
    for (int d = 0; d <= MAXD; d++) {
        lmr_d5 [d] = d / 5;
        lmr_d8 [d] = d / 8;
        lmr_d12[d] = d / 12;
    }
    for (int m = 0; m < MAX_MOVES; m++)
        lmr_mi8[m] = m / 8;
}

#define DEBUG_REDUCTION 0

#if DEBUG_REDUCTION
unsigned __int64 attempts=0, successes=0;
#endif

/*----------------------- Search function ---------------------*/
// Driver for the search process.  1st initialize important data
// structures, then do iterative deeping until time runs out.
move tree_search::search(position p, int time_limit, int T, game_rec *gr)
{
   char outstring[400], mstring[10];
   int g, done, pvi;
   int elapsed = 0, last_best_move = NOMOVE;
   int last_mate_score = 0, mate_iteration_count = 0; 
   int limit_search_depth = max_search_depth;
   position pv_pos; 
   move nomove; nomove.t = NOMOVE;

#if DEBUG_SEARCH
 search_outfile.open("search_debug.log");
#endif

  // Set searching flags in game record
  gr->searching = 1;
  gr->terminate_search = 0;

  // reseting the number of times the search time has been doubled
  time_double = 0;

  //------------------------------------------------------------------
  //  Log the current position in FEN format
  //------------------------------------------------------------------
  write_out("---------------------------------------------------------------\n");
  write_out("Starting search on position:\n");
  p.write_fen(1);

  //------------------------------------------------------------------
  //  Initialize variables and position for search 
  //------------------------------------------------------------------

  // Set start of search depth
  //  -- may be modified below as a result of successful
  //     pondering or play along the expected pv
  start_depth = 1;
  turn = T;                  // set game turn for tree search
  root_wtm = p.wtm;          // keep track of side-to-move at root 

  // be sure the move generation at the root 
  // does not give high scores to hmove,cmove,rmove,pmove
  p.hmove.t = NOMOVE;
  p.cmove.t = NOMOVE;
  p.rmove.t = NOMOVE;
 
  // Change hash ID to age old hash entries
  h_id++;
  if(h_id > 31) h_id = 1;

  //----------------------------------------------------------------------
  // Setup Pondering
  // - Find a ponder move (book, principal continuation, or short search)
  // - Make the move and set parameters for pondering
  //----------------------------------------------------------------------
  if(ponder) {
    if(gr->book && !tdata[0].pc[0][1].t) ponder_move = opening_book(p.hcode, p, gr);
    else ponder_move = tdata[0].pc[0][1];
    // if there is no ponder move, find one with a quick search
    if(!ponder_move.t) {
      write_out("Looking for a ponder move...\n");
      // mark singular move as not singular
      singular_response.t = NOMOVE;
      // initialize each thread
      for(int ti=0; ti<THREADS; ti++) {
	tdata[ti].n[0].pos = p;               // set the root node of the search
	tdata[ti].n[0].pos.hmove.t = NOMOVE;  // clear the expected best move from the position
	tdata[ti].init_thread_data(1);         // initialize data for thread
	tdata[ti].pc[0][0].t = NOMOVE;
#if NNUE
	if(nnue_available) {
	  tdata[ti].n[0].acc.dirty[0] = tdata[ti].n[0].acc.dirty[1] = true;
	  nnue_init_accumulator(tdata[ti].n[0].acc, p);
	}
#endif
      }
      // check TB at root
#if TABLEBASES
      if(p.pieces[0] + p.pieces[1]
	 + p.plist[0][PAWN][0] + p.plist[1][PAWN][0] <= EGTB) {
	root_tb_score = probe_tb(&p, 0);
      } else root_tb_score = -1;
#else 
      root_tb_score = -1;
#endif  /* TABLEBASES */
      max_ply = MAX(last_depth-7,3); 
      last_depth = max_ply;
      int save_post = post; post = 0;
      p.allmoves(&root_moves, &tdata[0]);
      assert(root_moves.count < MAX_MOVES);  
      sort_root_moves();
      g = search_threads(-MATE, +MATE, max_ply-1, THREADS);
      post = save_post;
      ponder_time = 0;
      ponder_move = tdata[0].pc[0][0];
      tdata[0].pc[0][0].t = NOMOVE;
      if(g == -TIME_FLAG) { ponder_move.t = NOMOVE; gr->searching = 0; return nomove; }
    }
    if(ponder_move.t) {
      // write move to a move string
      p.print_move(ponder_move, mstring, &(this->tdata[0]));
      if(!p.exec_move(ponder_move, 0)) { ponder_time = 0; gr->searching = 0; return nomove; }
      root_wtm = p.wtm;
      // don't search a stalemate or checkmate!
      if(p.in_check_mate() > 0) { ponder_time = 0; gr->searching = 0; return nomove; }
    } else { ponder_time = 0; gr->searching = 0; return nomove; }
    if(!ALLEG && post) {
      cout << "Hint: " << mstring << "\n";
      cout.flush();
    }
    snprintf(outstring, sizeof(outstring), "***** Ponder Move: %s *****\n", mstring);
    write_out(outstring);
    // add ponder move to position list for all threads
    for(int ti=0;ti < THREADS;ti++) { tdata[ti].plist[T-1] = p.hcode; }
    // set ponder time doubling count
    ponder_time_double = 0;
    // set target time limit to 1/4 the previous maximum allowed
    time_limit = max_limit/4;
  }

  //---------------------------------------------  
  // checking tablebases at root
  //---------------------------------------------
#if TABLEBASES
  if(p.pieces[0] + p.pieces[1]
      + p.plist[0][PAWN][0] + p.plist[1][PAWN][0] <= EGTB) {
   root_tb_score = probe_tb(&p, 0);
  } else root_tb_score = -1;
#else 
  root_tb_score = -1;
#endif  /* TABLEBASES */

  //-----------------------------  
  // checking book
  //-----------------------------  
  if(gr->book && !ponder) {
    bookm = opening_book(p.hcode, p, gr);
    if(bookm.t) {
     tdata[0].pc[0][0] = bookm; tdata[0].pc[0][1].t = 0;
     // output the book move to log or search screen
     p.print_move(bookm, mstring, &(this->tdata[0]));
     snprintf(outstring, sizeof(outstring), "Book Move Found: %s\n", mstring);
     write_out(outstring);
     cout.flush();
     gr->searching = 0; return tdata[0].pc[0][0];
    } else {
     no_book++;
     if(no_book >= 3) gr->book = 0;
    }
  }

  //------------------------------------------------------------------
  //   React appropriately based on whether we pondered previously
  //     --> or if we had a singular move!
  //------------------------------------------------------------------

  // if last search was a ponder and if user made expected move and
  // if ponder time is greater than limit, return best move
  if(last_ponder && ponder_time >= (time_limit<<ponder_time_double)
     && p.last.t == ponder_move.t && tdata[0].pc[0][0].t) {
    // output move to log or search screen
    p.print_move(tdata[0].pc[0][0], mstring, &(this->tdata[0]));
    snprintf(outstring, sizeof(outstring), "Guessed Last Move! Returning: %s\n", mstring);
    write_out(outstring);
    cout.flush();
    singular_response.t = NOMOVE;
    gr->searching = 0; return tdata[0].pc[0][0];
  }

  // if last search expected a singular move after opponents response,
  //  move instantly
  if(!ponder && p.last.t && singular_response.t 
     && ((p.last.t == ponder_move.t && last_ponder && singular_response.t==tdata[0].pc[0][0].t)
		  || (p.last.t == tdata[0].pc[0][1].t && singular_response.t==tdata[0].pc[0][2].t && !last_ponder))) {
    // output move to log or search screen
    tdata[0].pc[0][0] = singular_response;
    p.print_move(tdata[0].pc[0][0], mstring, &(this->tdata[0]));
    snprintf(outstring, sizeof(outstring), "Singular Reply Move! Returning: %s\n", mstring);
    write_out(outstring);
    cout.flush();
    singular_response.t = NOMOVE;
    gr->searching = 0; return tdata[0].pc[0][0];
  } else if(!ponder) { singular_response.t = NOMOVE; }

  // if last search was a ponder, user made the expected move, but we
  // didn't search deep enough, continue search
  if(last_ponder && p.last.t == ponder_move.t && tdata[0].pc[0][0].t) {
    start_depth = MAX(last_depth-1,1);
    h_id--; // reset hash id to mark entries as current
    write_out("Guessed last move, but searching deeper...\n");
    cout.flush();
    // set target time
    limit = (time_limit<<ponder_time_double) - ponder_time;
  } 
  // else if we are in analysis mode or pondering we can keep  
  //   searching at the same depth if the user made the pv move.
  else if((analysis_mode && tdata[0].pc[0][0].t == p.last.t && tdata[0].pc[0][0].t) || ponder) {
    start_depth = MAX(last_depth-1, 1);
    tdata[0].pc[0][0].t = NOMOVE;
    limit = time_limit;
  }
  // else if we had no pondering but are still in the pv,
  //   we can make use of the results of the previous 
  //   search to speed things up...  but we must have a
  //   valid move to search first at this move, so tdata[0].pc[0][2] must
  //   have a move in it
  else if(tdata[0].pc[0][1].t == p.last.t && tdata[0].pc[0][1].t && !last_ponder) {
    if(tdata[0].pc[0][2].t) start_depth = MAX(last_depth-2, 1);
    tdata[0].pc[0][0].t = NOMOVE;
    limit = time_limit;
  }
  // else just clear the first move in the pv
  //  and set the target time
  else {  
    tdata[0].pc[0][0].t = NOMOVE;     // clear first move in the pv
    limit = time_limit;      // set target time
  }

  //------------------------------------------------------------------
  //   Initialize search loop
  //------------------------------------------------------------------

  start_depth = 1;  // always force start_depth = 1 to test if this has been a problem

  // initialize starting position, best move, and search data in all threads
  for(int ti=0;ti<THREADS;ti++) {
    tdata[ti].n[0].pos = p;               // set the root node of the search
    tdata[ti].n[0].pos.hmove.t = NOMOVE;  // clear the expected best move from the position
    tdata[ti].init_thread_data(ponder);
#if NNUE
    if(nnue_available) {
      tdata[ti].n[0].acc.dirty[0] = tdata[ti].n[0].acc.dirty[1] = true;
      nnue_init_accumulator(tdata[ti].n[0].acc, p);
    }
    if(!ti) tdata[0].n[0].premove_score = p.score_pos(gr,tdata,
                                           nnue_available ? &tdata[0].n[0].acc : nullptr);
#else
    if(!ti) tdata[0].n[0].premove_score = p.score_pos(gr,tdata);
#endif
    if(ti > 0) {
      tdata[ti].n[0].premove_score = tdata[0].n[0].premove_score;
    }
  }

  last_ponder = 0;
  ponder_time = 0;
  max_limit = int(MIN(8.0*time_limit, MAX(time_limit, gr->timeleft[p.wtm]/4.0)));
  if(!gr->mttc && !xboard && !tsuite && !analysis_mode) {max_limit = int(gr->timeleft[p.wtm]);}
  max_limit = MIN(max_limit, max_search_time*100);
  if(!tsuite && !analysis_mode) {
    if(!gr->mttc && !xboard) {
      limit = MIN(max_limit, limit);
    } else {
      limit = MIN(max_limit/2, limit);
    }
    if(gr->timeleft[p.wtm] < 1000) {
      start_depth = 1;
    }
  }

  // Attempt to avoid problems with interface time lags when short on time
  //  Won't kick in unless we have < gr->base/10 sec left.
  //  Minimum time cushion is the smaller of 6 seconds or gr->base/100, but 
  //  it can be larger depending on lag and moves to go until time control.  
  //  NOTE: gr->base is in seconds, all times below are in centi-seconds
  int min_cushion = MAX(MIN((int)gr->base, 600), 3);  // at least 3 cs for increment-only games
  int time_cushion = MAX((gr->mttc+3)*average_lag, min_cushion);
  int max_margin = MAX(MIN((int)gr->base*10, 6000), 3*min_cushion);  // non-zero for increment-only games
  if(xboard && gr->timeleft[p.wtm] <= MIN(3*time_cushion, max_margin)) {
    limit = MIN(limit, MAX(1,gr->timeleft[p.wtm]-2*time_cushion));
    max_limit = limit;  // no time extensions in this mode
    write_out("Short time control restrictions active.\n");
  }


  start_time = GetTime();

  last_depth = 1;
  g_last = 0;
  id_score_count = 0;

  //------------------------------------------------------------------
  //  Generate a root move list
  //------------------------------------------------------------------
  p.allmoves(&root_moves, &tdata[0]);
  assert(root_moves.count < MAX_MOVES);  

  // Set limiting search depth in case of training mode
#if TRAINING_MODE
  if(max_search_depth < MAXD && p.gstage > 6) {
    limit_search_depth = max_search_depth*p.gstage/6;
    if(limit_search_depth > MAXD) limit_search_depth = MAXD;
  }
#endif

  //------------------------------------------------------------------
  //   Log the search conditions
  //------------------------------------------------------------------   
  snprintf(outstring, sizeof(outstring), "Target search time: %i, max %i, timeleft (%i %i)\n",
	  limit, max_limit, int(gr->timeleft[0]), int(gr->timeleft[1]));
  write_out(outstring);
  snprintf(outstring, sizeof(outstring), "Starting depth: %i\n", start_depth); 
  write_out(outstring);

  //------------------------------------------------------------------
  //
  // Main search loop for iterative deeping
  //  - The iteration limit of 127 is set by hash depth storage limit
  //
  //------------------------------------------------------------------
  for(max_ply = start_depth; max_ply < MIN(MAX_MAIN_TREE,limit_search_depth+1); max_ply++)
  {

    // Sort root moves for the new iteration
    sort_root_moves();

    //------------------------------------------
    // choose alpha/beta limits for root search
    //  -- only narrow window after a few 
    //     iterations
    //  -- NOTE: window size and how it changes
    //     with iteration and number of fails 
    //     strongly interacts with time extension
    //     mechanisms!!
    //------------------------------------------
    if(max_ply > MAX(start_depth,3))
     { root_alpha = g_last-15; root_beta = g_last+15; }
    else
     { root_alpha = -MATE; root_beta = +MATE; }

#if USE_MTD
    /* use MTD(f) algorithm with granular eval */
    int upper_bound = MATE, lower_bound =-MATE, beta, up = -2, down = -1;
    g = g_last;
    while(upper_bound > lower_bound) {
      if(g==lower_bound) beta=MIN(g+EVAL_GRAN*MAX(down,1),upper_bound);
      else beta = MAX(g-EVAL_GRAN*MAX(up,0), lower_bound); 
      g = search_threads(beta-EVAL_GRAN, beta, max_ply-1-fail_high, THREADS);
      //      g = tdata[0].n[0].root_pvs(beta-EVAL_GRAN, beta, max_ply-1);
      if(g == -TIME_FLAG) break;
      if(g < beta) { 
	upper_bound = g;
	up++;
	down=-1;
      } else { 
	lower_bound = g;
        down++;
	up=-2;
      }
    }
#else    
   /* use PVS Algorithm with aspiration windows */
    //------------------------------------------
    // loop to call root search with increasing
    //   alpha/beta limits, will break if
    //   - fail high and fail low happen sequentially
    //   - score (g) is between alpha/beta limits
    //   - uses fail_high reduction trick reported
    //     by Robert Houdart on CCC to resolve fail highs
    //     more quickly
    //------------------------------------------
    fail_low = 0; fail_high = 0;
    while(1) {
      g = search_threads(root_alpha, root_beta, max_ply-1-fail_high, THREADS);
      if(g == -TIME_FLAG) break;
      if(g <= root_alpha && !fail_high) {
	root_beta = root_alpha; fail_low = 1;
	root_alpha = MAX(-MATE,g+1.5*(root_alpha-g_last));
      } else if(g >= root_beta && !fail_low) { 
	root_alpha = root_beta; fail_high = 1;
	// resort root moves if we changed our mind
        //  about the best move.
	if(tdata[0].pc[0][0].t != root_moves.mv[0].m.t) {
	  sort_root_moves();
	}
	root_beta = MIN(+MATE,g+1.5*(root_beta-g_last));
      } else break;
      if(root_alpha < MAX(-5000,g_last-500)) root_alpha = -MATE;
      if(root_beta > MIN(5000,g_last+500)) root_beta = +MATE;
    }
#endif

    // display results of this search iteration
    if(g != -TIME_FLAG) {
      if(post && !xboard) search_display(g);
      log_search(g);
      best_depth = max_ply;
    }

    // check total time used in this search
    elapsed = GetTime() - start_time;

    // if time is up OR we are interrupted... break
    //  -- require at least 3 ply before breaking due to time being up
    //     This is important to avoid draw by reps!
    if(g == -TIME_FLAG || inter() || (elapsed >= limit && !ponder && max_ply > 3)) { 
      break;
    }

    // Set CHECK_INTER integer for how often to check for
    //  time elapsed
    if(elapsed > 0 && limit > 0) {
      int min_interval = (limit*tdata[0].node_count)/(10*elapsed);
      for(CHECK_INTER = 128; CHECK_INTER < 65536; CHECK_INTER *=2) {
	if(CHECK_INTER > min_interval) {
	  CHECK_INTER /= 2;
	  break;
	}
      }
      CHECK_INTER -= 1;
    }

    g_last = g;
    // push this depth's score into the ID history ring (oldest dropped when full)
    if (id_score_count < TD_ID_HIST) id_score_count++;
    for (int _i = 0; _i < id_score_count - 1; _i++)
        id_scores[_i] = id_scores[_i + 1];
    id_scores[id_score_count - 1] = g;
    last_depth = max_ply;
    last_best_move = tdata[0].pc[0][0].t;

    // keep track of how many iterations we've had the
    // same mate score for
    if(g > (MATE-1000) || g < -(MATE-1000)) {
      if(g == last_mate_score) {
	mate_iteration_count++;
      }
      last_mate_score = g;
    }

    // break if we have a draw or a consistent mate score
    if((g == 0 && max_ply > MAXD-3) || (mate_iteration_count > 1)) break;

    // if we have a root tb hit and a move, break after third iter.
    if(root_tb_score > -1 && tdata[0].pc[0][0].t && max_ply > 2) break;

  }

  //------------------------------------------------------------------
  //  Clean up after search, record time used and post/log results
  //------------------------------------------------------------------

  // record the ponder state
  if(ponder) {
   last_ponder = 1; ponder_time = elapsed;
  }

  // These are a collection of counters that keep track of search
  // statistics and timing checks.
  unsigned __int64 node_count=0ULL, eval_count=0ULL, extensions=0ULL, qchecks=0ULL;
  unsigned __int64 phash_count=0ULL, hash_count=0ULL, hmove_count=0ULL, q_count=0ULL;
  unsigned __int64 null_cutoff=0ULL, internal_iter=0ULL, egtb_probes=0ULL, egtb_hits=0ULL;
  unsigned __int64 failed_high=0ULL, first_fail_high=0ULL, shash_count=0ULL, sing_count = 0ULL;

  // sum values over all threads
  for(int t=0;t<THREADS;t++) {
    node_count+=tdata[t].node_count;
    eval_count+=tdata[t].eval_count;
    extensions+=tdata[t].extensions;
    qchecks+=tdata[t].qchecks;
    phash_count+=tdata[t].phash_count;
    hash_count+=tdata[t].hash_count;
    hmove_count+=tdata[t].hmove_count;
    q_count+=tdata[t].q_count;
    null_cutoff+=tdata[t].null_cutoff;
    internal_iter+=tdata[t].internal_iter;
    egtb_probes+=tdata[t].egtb_probes;
    egtb_hits+=tdata[t].egtb_hits;
    failed_high+=tdata[t].fail_high;
    first_fail_high+=tdata[t].first_fail_high;
    shash_count+=tdata[t].shash_count;
    sing_count+=tdata[t].sing_count;
  }

  // prevent divides by zero
  if(failed_high < 1) failed_high = 1;
  if(elapsed < 1) elapsed = 1;

  if(!xboard && !ALLEG && post) {
   cout << "\nnodes = " << node_count
        << " hash moves = " << hmove_count
        << " qnodes = " << q_count
        << " evals = " << eval_count << "\n";
   cout << "hash hits = " << hash_count
        << " pawn hash hits = " << phash_count 
	<< " score hash hits = " << shash_count << "\n";
   cout << "node_rate = " << int(100.0*(float(node_count)/float(elapsed)))
        << " null cuts = " << null_cutoff
        << " exten = " << extensions
        << " sing_ext = " << sing_count
        << " qchecks = " << qchecks << "\n";
   cout << "int_iter = " << internal_iter
        << " egtb_probes = " << egtb_probes
        << " egtb_hits = " << egtb_hits
        << " fail_high(%) = " << float(int(10000*(float(first_fail_high)/float(failed_high))))/100.0 << "\n";
#if DEBUG_REDUCTION
   cout << " attempts = " << attempts << " successes = " << successes << "\n";
#endif
  }

  snprintf(outstring, sizeof(outstring), "nodes = %llu, qnodes = %llu, evals = %llu, hash moves = %llu\n",
	  node_count, q_count, eval_count, hmove_count);
  write_out(outstring);
  snprintf(outstring, sizeof(outstring), "hash hits = %llu, pawn hash hits = %llu, score hash hits = %llu\n",
	  hash_count, phash_count, shash_count);
  write_out(outstring);
  snprintf(outstring, sizeof(outstring), "node_rate = %i, null_cuts = %llu, exten = %llu, qchecks = %llu, int_iter = %llu\n",
	  int(100.0*(float(node_count)/float(elapsed))), null_cutoff, extensions, qchecks,
	  internal_iter);
  write_out(outstring);
  snprintf(outstring, sizeof(outstring), "egtb_probes = %llu, egtb_hits = %llu, fail_high(pct) = %5.2f\n",
	  egtb_probes, egtb_hits, 100.0*(float(first_fail_high)/float(failed_high)));
  write_out(outstring);

  p.print_move(tdata[0].pc[0][0], mstring, &(this->tdata[0]));

  if(ponder) { 
    snprintf(outstring, sizeof(outstring), "***** Pondering ended ***** Time: %i, Time required doublings: %i\n",
	    ponder_time, ponder_time_double);
    write_out(outstring);
  } else {
    snprintf(outstring, sizeof(outstring), "Return move from search: %s\n", mstring);
    write_out(outstring);
  }
  write_out("---------------------------------------------------------------\n");

#if DEBUG_SEARCH
 search_outfile.close();
#endif

  //------------------------------------------------------------------
  //   Do Book learning if score is right
  //------------------------------------------------------------------

  // book learning if applicable
  if(!ponder && gr->learn_bk && BOOK_LEARNING) {
    if(!gr->book && gr->learn_count > 0 && wbest > +LEARN_SCORE)
      { book_learn(1, gr); gr->learned = gr->learn_count; gr->learn_count = -1; }
    else if(!gr->book && gr->learn_count > 0 && wbest < -LEARN_SCORE)
      { book_learn(0, gr); gr->learn_count = -1; }
    else if(gr->learned && wbest < -LEARN_SCORE)
      { gr->learn_count = gr->learned; book_learn(-1, gr);
        gr->learned = 0; gr->learn_count = -1; }
   }

  //-----------------------------------------
  // Assign a singular move, if appropriate
  //-----------------------------------------
  if((tdata[0].pc[0][2].b.type&SINGULAR) && !ponder) {
    tdata[0].pc[0][2].b.type &= NOT_SINGULAR;
    singular_response=tdata[0].pc[0][2];
  } else if(!ponder) singular_response.t = NOMOVE;
  else tdata[0].pc[0][2].b.type &= NOT_SINGULAR;

  //------------------------------------------------------------------
  //   Return the best move found by the search
  //------------------------------------------------------------------
  gr->searching = 0; return tdata[0].pc[0][0];

}

/* Function to update principle continuation */
// The principle continuation is stored as a "triangular" array.
// It is updated by doing a mem-copy of the principle continuation
// found at deeper depths to this depth + the move at this depth
// is stuffed in first.
inline void ts_thread_data::pc_update(move pcmove, int ply)
{
 pc[ply][ply].t = pcmove.t;
 for (int pci = ply+1; pci < MAXD; pci++)
 {
   if(!(pc[ply+1][pci].t)) { 
     pc[ply][pci].t = NOMOVE;
     break; 
   }
   pc[ply][pci].t = pc[ply+1][pci].t;
 }
}

// Special sort to shuffle best move to the top of the list
// -- assumes there is a open space in array above the
//    upperbound location
inline void MSort(move_rec *Lb, move_rec *Ub)
{
  move_rec *I, *J;

   J = Lb;
   for(I = Lb+1; I <= Ub; I++) {
       if (I->score > J->score) { J = I; }
   }
   if(J != Lb) {
     *(Ub+1) = *Lb;
     *Lb = *J;
     *J = *(Ub+1);
   }

   assert(Ub-Lb < MAX_MOVES-1);
}

// Sort for root move list
void tree_search::sort_root_moves() {

  int best_move = tdata[0].pc[0][0].t;

  // best move should have been decided, but if not, check hash table  
  if(!best_move) { 
    best_move = get_move(&tdata[0].n[0].pos.hcode); 
  }

  // Sift through root list to find best move
  //  also aggregate history scores at same time
  for(int mi = 0; mi < root_moves.count; mi++) {
    if(root_moves.mv[mi].m.t == best_move) {
      move_rec temp_move = root_moves.mv[0];
      root_moves.mv[0] = root_moves.mv[mi];
      root_moves.mv[mi] = temp_move;
    }
    // assign a history score to those non-capture/non-killer/non-hash moves
    if(root_moves.mv[mi].score < 2000000) {
      root_moves.mv[mi].score = 0;
      for(int ti=0; ti < THREADS; ti++) {
        root_moves.mv[mi].score += tdata[ti].history[tdata[0].n[0].pos.sq[root_moves.mv[mi].m.b.from]][root_moves.mv[mi].m.b.to];
      }
      root_moves.mv[mi].score /= THREADS;
    }
  }

  // Sort the remaining root moves
  int mi = 1;
  if(!best_move) mi = 0;
  for(; mi < root_moves.count-1; mi++) {
    MSort(&root_moves.mv[mi],&root_moves.mv[root_moves.count-1]);
  }  

}


/*-------------------- Principle variation function --------------------*/
// --> This one is specific to the root of the search tree
//--------------------------------------------------------------------------
// This function uses the improvement on the
// alpha-beta scheme called "principle variation search".
// After the first node, which is searched with the full alpha-beta window,
// the rest of the nodes are searched with a "null" window: alpha,alpha+1.
//--------------------------------------------------------------------------

void search_node::root_pvs()
{

  int alpha = tdata->alpha;
  int beta = tdata->beta;
  int depth = tdata->depth;

 // increase node count
 tdata->node_count++;

#if DEBUG_SEARCH
 // Debug archiving of search to a file
 char space_string[50] = "\0";
 char last_move_string[10];
 search_outfile << space_string << "->Ply: " << ply << ", max_ply: "
	 << ts->max_ply << ", depth: " << depth << ", alpha: " << alpha
	 << ", beta: " << beta << ", last: " << last_move_string << "\n";
#endif


 // ----------------------------------------------------------
 //    Do search preliminaries
 // ----------------------------------------------------------
 // initialize some variables
 best = -MATE+ply;
 save_alpha = alpha;
 null_hash = 1;  // a precaution, but there are no null move in PV nodes
 pos.threat_square = 64; 
 singular = 0;
 pv_node = 1;
 depth_mod = 0;
 pos.hmove.t = ts->root_moves.mv[0].m.t;
 int next_best_index = 0;
 
 // --------------------------------------
 //   Singular extension test
 // --------------------------------------

 #define ROOT_SMARGIN  25
 
 if(depth > 7
    && pos.hmove.t
    && pos.pieces[pos.wtm] > 1
    && alpha > -MATE/2
    // not a previous fail highs or lows
    //   at this depth
    && !ts->fail_low
    && !ts->fail_high
    && ts->root_moves.count > 1
    && ply + depth < MAXD-3) {

    next->pos = pos; next->pos.last.t = NOMOVE;
    next->premove_score = premove_score;
#if NNUE
    if(nnue_available) { next->acc = acc; next->acc.computed = true; }
#endif
#if DEBUG_SEARCH
    search_outfile << space_string << "Entering an Singular Ext. loop\n";
#endif
    int test_score = ts->g_last-ROOT_SMARGIN;
    int test_depth = depth/2;
    score = next->pvs(test_score-1,test_score, test_depth, 1, pos.hmove.t);
    if(score == -TIME_FLAG) { tdata->g = -TIME_FLAG; return; }
#if DEBUG_SEARCH
    search_outfile << space_string << "Leaving an Singular Ext. loop\n";
#endif
    // if score < test_score -- the first move is singular
    if(score < test_score) {
      singular = 1;
    } else if(next->pos.hmove.t) {
      // identify the move that proved this was not singular
      for(mi = 1; mi < ts->root_moves.count; mi++) {
	if(next->pos.hmove.t == ts->root_moves.mv[mi].m.t) {
	  next_best_index = mi;
	  break;
	}
      }
    }

 } // closing bracket of singular search code

 // --------------------------------------------------------
 //     Move Loop
 //     Basic plan is as follows...
 //     -- loop over each move from the list
 //     -- make the move to the position in the "next" node
 //     -- do any depth reductions
 //     -- search the "next" node to depth-UNIT_DEPTH
 //     -- use score from move and alpha/beta cuts
 //        to decide if we are done (fail high) or
 //        need to continue... do associated book
 //        keeping
 // --------------------------------------------------------
 mcount = 0, first = 1; skipped = 0;
 moves.count = ts->root_moves.count; // needed for qsearch check
 while (mcount < ts->root_moves.count+skipped && best < beta)
 {

   // assign the move index, first see if we are in
   //   skipped moves at the end of the list
   //   due to those moves being searched in other
   //   threads   
   if(mcount >= ts->root_moves.count) {
     mi = skip_move[mcount-ts->root_moves.count];
   // else assign the move index normally
   } else {
     mi = mcount;
   }

   //-----------------------------------------------
   // Assign the move to be searched to a short cut
   //-----------------------------------------------
   smove = ts->root_moves.mv[mi].m;
   
   //-----------------------------------------------
   // Loop over other the share data for other 
   //  threads to see if this move is being
   //  searched by another and should be skipped
   //  for the time being.  The move will be 
   //  added to a list to be visited at the end
   //  of the movelist.
   //-----------------------------------------------
   if(THREADS > 1 && !first && skipped < 23 && mcount < ts->root_moves.count) {
     int skip = 0;
     for(int ti = 0; ti < THREADS; ti++) {
       if(ti == tdata->ID) continue;
       if(ts->share_data[ply][ti].check_active_move(pos.hcode,depth,save_alpha,beta,smove.t)) {
	 skip = 1; 
	 break;
       }
     }
     if(skip) {
       skip_move[skipped] = mi;
       skipped++;
       mcount++;
       continue;
     }
   }
 
   // -------------------------------------------------------------
   // Allow possibility for an oversight when knowledge_scale < 50
   // -------------------------------------------------------------
   if(gr->knowledge_scale < 50 && !first) {
     if((((pos.hcode^(smove.b.to+smove.b.from))&63ULL) > gr->knowledge_scale)
	&& (((pos.hcode^smove.b.to^smove.b.from)&3)==3)) {
       mcount++;
       continue;
     }
   }

   // -------------------------------------------------
   // making move in the next position in the search
   //   -- if it is illegal, skip to next move in list
   // -------------------------------------------------
   next->pos = pos;
   if(!next->pos.exec_move(smove, ply)) {
    mcount++;
    continue;
   }
#if NNUE
   if(nnue_available) nnue_record_delta(next->acc, pos, next->pos, smove);
#endif

   //---------------------------------------------
   // now set this move as the one being worked
   //---------------------------------------------
   if(THREADS > 1) {
     ts->share_data[ply][tdata->ID].set_active_move(pos.hcode,depth,save_alpha,beta,smove.t);
   }

   //---------------------------------------------
   // ensure we have a move to play at all times
   //---------------------------------------------
   if(!tdata->pc[0][0].t) {
     tdata->pc[0][0].t = smove.t;
   }
 
#if DEBUG_SEARCH
   pos.print_move(smove, last_move_string, tdata);
   search_outfile << space_string << "Move: " << last_move_string;
   search_outfile << " Score: " << moves.mv[mi].score << "\n"; 
#endif

   // -------------------------------------------------------
   // initalize depth modifications for this move to be zero    
   // -------------------------------------------------------
   depth_mod = 0; 

   //---------------------------------------------------
   // if we caused a check, extend if depth is small
   //  OR if move is singular
   //---------------------------------------------------
   if(!mi && singular) 
     { depth_mod = 1; tdata->extensions++; tdata->sing_count++; }
   else if(depth < 3 && next->pos.check) 
     { depth_mod = 1; tdata->extensions++; }

   // -----------------------------------------------------
   //   Move Reductions at Ply 0
   //    -- only if current best score is high enough
   //       relative to the score in last iteration
   //    -- position is scored with a search of reduced
   //       depth to decide if a reduction should happen
   // -----------------------------------------------------
   if(depth > 4 && !pos.check && best > ts->g_last-NO_ROOT_LMR_SCORE
      && !depth_mod && !first && !next->pos.check 
      && mi != next_best_index
      && (ts->root_moves.mv[mi].score < 2000000)    // not a winning capture, promotion, killer move, etc...
      && best > -(MATE/2)) {                   // best > -(MATE/2) important!
     //--------------------
     // Reduce by one ply
     //--------------------
     depth_mod -= 1;
     //------------------------------------------------------------------
     // make another reduction decision based on history or position in move list
     // --> reduce if this move loses historically (~ 90% fail low history)
     //------------------------------------------------------------------
     if(tdata->history[pos.sq[smove.b.from]][smove.b.to] < -10*(depth)) {
       depth_mod -= 1 + lmr_d12[depth];
       if(tdata->history[pos.sq[smove.b.from]][smove.b.to] < -23*(depth)) {
	 depth_mod -= 1 + lmr_d8[depth];
       }
     }
   }

   // -----------------------------------
   // initialize tb_hit variable 
   // -----------------------------------
   tdata->tb_hit = 0;

   // -----------------------------------
   //   Search the next node in the tree
   // -----------------------------------
   if(depth+depth_mod < 1) {
     score = -next->qsearch(-beta, -alpha, 0);
     if (score == TIME_FLAG) { tdata->g = -TIME_FLAG; return; }
     if(tdata->tb_hit && ts->root_tb_score > (MATE/2) && score <= ts->root_tb_score) score = alpha;
   } else {
    if(first) {
      score = -next->pvs(-beta, -alpha, depth+depth_mod-1, 1, 0);
      if (score == TIME_FLAG) { tdata->g = -TIME_FLAG; return; }
      if(tdata->tb_hit && ts->root_tb_score > (MATE/2) && score <= ts->root_tb_score) score = alpha;
    } else {
      score = -next->pvs(-alpha-1, -alpha, depth+depth_mod-1, 0, 0);
      if (score == TIME_FLAG) { tdata->g = -TIME_FLAG; return; }
      if(tdata->tb_hit && ts->root_tb_score > (MATE/2) && score <= ts->root_tb_score) score = alpha;
      if (score > alpha && score < (MATE/2)) { 
	if(depth_mod < 0) depth_mod = 0;  // for checking move reductions on re-search
	score = -next->pvs(-beta, -alpha, depth+depth_mod-1, 1, 0);
	if (score == TIME_FLAG) { tdata->g = -TIME_FLAG; return; }
      }
    }
   }

#if DEBUG_SEARCH
 search_outfile << space_string << "Returned value: " << score << "\n";
#endif

   // ----------------------------------------
   // Handle a first move fail low
   // ----------------------------------------
   if(first && score <= alpha) { 
     // ----------------------------------------
     // Panic mode for additional time
     //  -- also interacts with expanding fail 
     //     low window to allow up to 3 extensions
     //     if a move fails low three times
     //  -- only in the main thread (ID == 0)
     //-----------------------------------------
     if(!tdata->ID && ts->limit < ts->max_limit/2 
	&& score <= ts->g_last-EXTEND_TIME_SCORE*(ts->time_double+1)
	&& !ts->tsuite 
	&& GetTime() - ts->start_time >= ts->limit/4) {
       if(!ts->ponder) { 
	 ts->limit = MIN(2*ts->limit, ts->max_limit/2);
	 if(logging) logfile << "Extending search to: " << ts->limit << "\n";
       } else if(ts->ponder_time_double < 2) {
	 ts->ponder_time_double++;
	 if(logging) logfile << "Doubling required ponder time\n";
       }
       ts->time_double++;
     }
     // set the PV moves appropriately
     tdata->pc[0][1].t = next->pos.hmove.t;
     tdata->pc[0][2].t = NOMOVE;
     // log/display a fail low when it happens
     //  -- only in the main thread (ID == 0)
     if(!tdata->ID) {
       tdata->fail = -1;
       if(post && !xboard) ts->search_display(score);
       ts->log_search(score);
       tdata->fail = 0;
     }
   }

   //--------------------------------------------------------------
   // possibly decrease time if appropriate, only in first thread
   //--------------------------------------------------------------
   if(first && singular && score > alpha) {
     if(!tdata->ID && ts->limit >= ts->max_limit/16 
	&& !ts->tsuite 
	&& GetTime() - ts->start_time >= ts->limit/4) {
       if(!ts->ponder) { 
	 ts->limit /= 2;
	 if(logging) logfile << "Reducing search to: " << ts->limit << "\n";
       } else if(ts->ponder_time_double > 0) {
	 ts->ponder_time_double--;
	 if(logging) logfile << "Halving required ponder time\n";
       }
     }
   }

   // ------------------------------------ 
   //   Update best score
   // ------------------------------------ 
   if(score > best) { 
     best = score;
   }

   // ----------------------------------------------- 
   //   If score is not a fail-low
   //     -- increase alpha bound
   //     -- store best move
   //     -- update pv and history/killer move data
   //     -- if root-node, display line
   //   Else update history score for move
   // ----------------------------------------------- 
   if (score > alpha) {
     alpha = score;
     pos.hmove = smove;
     //---------------------------------------------
     // if score >= beta, increase fail high counts
     //---------------------------------------------
     if(score >= beta) {
       tdata->fail_high++;
       if(!mi) tdata->first_fail_high++;
       tdata->fail = 1;
       tdata->pc[0][0].t = smove.t; 
       tdata->pc[0][1].t = next->pos.hmove.t;  // guess at best reply
       tdata->pc[0][2].t = NOMOVE;
     } else tdata->pc_update(smove, ply);
     //---------------------------------
     // display/output search info
     //---------------------------------
     //ts->wbest = score; ts->wply = ts->max_ply;  // whisper variables
     // only display search in the default thread
     if(tdata->ID==0) {
       if(post && !xboard && (!tdata->fail || !first)) ts->search_display(score);
       ts->log_search(score);
     }
     tdata->fail = 0;
   } 

   // ------------------------------------ 
   // update move loop variables
   // ------------------------------------    
   first = 0;
   mcount++;

 }  /* End of move loop */     

 //---------------------------------------------
 // Clear active move data for shared work
 //---------------------------------------------
 if(THREADS > 1) { 
   ts->share_data[ply][tdata->ID].set_active_move(0,0,0,0,0);
 }

 // ----------------------------------------------------------------- 
 // if it is mate or stalemate; no move is in principle continuation
 // ----------------------------------------------------------------- 
 if(first && mcount == ts->root_moves.count) {
   if(!pos.check) best = 0;        // score is even if stalemate
   pos.hmove.t = NOMOVE;
   tdata->pc[0][0].t = NOMOVE;
 } 

 // ------------------------------------------------------ 
 // storing position in the hash table
 //  -- assign the hash move to be the best move if the
 //     position is a fail-low with no best move
 //  -- note the double store for the PV node
 //     to guarantee these are available to zero-window
 //     nodes as well. 
 //  -- To further explain, the two types of nodes are
 //     stored separately (just using a different hash
 //     signature) because the searches can be very
 //     different
 // ------------------------------------------------------ 
 if(best != 0 || save_alpha == 0) {   // don't store draws depending on path
   put_hash(&pos.hcode, best, save_alpha, beta, depth, pos.hmove.t, ts->h_id, ply); 
 }

 tdata->g = best;
 
 return;

}

/*-------------------- Principle variation function ------------------------------*/
// This function uses the improvement on the
// alpha-beta scheme called "principle variation search".
// After the first node, which is searched with the full alpha-beta window,
// the rest of the nodes are searched with a "null" window: alpha,alpha+1.
//--------------------------------------------------------------------------

int search_node::pvs(int alpha, int beta, int depth, int in_pv, int move_to_skip)
{

 // increase node counts
 tdata->node_count++;
 
 // ----------------------------------------------------------
 //     Check search interrupt... defined in search.h
 // ----------------------------------------------------------
 SEARCH_INTERRUPT_CHECK();

 //------------------------------
 // Don't search too deep!!
 //------------------------------
 if(ply >= MAX_MAIN_TREE) {
   return qsearch(alpha,beta,0);
 }

#if DEBUG_SEARCH
 // Debug archiving of search to a file
 char space_string[50] = "\0";
 char last_move_string[10];
 for(int si = 0; si < ply; si++) strcat(space_string," ");
 if(ply) prev->pos.print_move(pos.last, last_move_string, tdata);

 search_outfile << space_string << "->Ply: " << ply << ", max_ply: "
	 << ts->max_ply << ", depth: " << depth << ", alpha: " << alpha
	 << ", beta: " << beta << ", last: " << last_move_string << "\n";

#endif

 // ----------------------------------------------------------
 //    Do search preliminaries
 //       -- setup variables
 //       -- check for time interrupt of search
 //       -- test fifty move rule and 3-rep
 //       -- look in the hash table for a cut-off or a move
 //       -- look in the end-game-tablebases (egtb)
 // ----------------------------------------------------------

 // initialize some variables
 best = -MATE+ply;
 save_alpha = alpha;
 null_hash = 1;  
 hflag = 0;
 pv_node = in_pv;
 moves.count = 0;
 depth_mod = 0;
 singular = 0;
 
 // preserve threat square on IID, but
 //  otherwise reset -- note doesn't affect
 //  NULL move which already has threat_square
 //  reset
 if(pos.last.t) { 
   pos.threat_square = 64; 
   pos.threat_check = 0;
 }
 
 // zero expected PV move in this position
 pos.hmove.t = NOMOVE;
 tdata->pc[ply][ply].t = NOMOVE;

 //--------------------------------------------------
 //    Add hash code for this position to position list
 //--------------------------------------------------
 tdata->plist[ts->turn+ply-1] = pos.hcode;

 //--------------------------------------------------
 //    Do Draw Testing
 //--------------------------------------------------
 if(pos.last.t) {	   
   // fifty move rule
   if(pos.fifty >= 100) {
     //return 0;
     return(MAX(MIN(0,beta),alpha));
   }
   // avoid repeating a position if possible
   int test_ply = ply;
   for(int ri = ts->turn+test_ply-3; ri >= ts->turn+test_ply-pos.fifty-1; ri -= 2) {
     // account for IID or singular tests in move sequence
     while(ri > 0 && tdata->plist[ri] == tdata->plist[ri-1]) { ri--; test_ply--; }
     // check code for rep.
     if(tdata->plist[ri] == pos.hcode) {
       //return 0;
       return(MAX(MIN(0,beta),alpha));
     }
   }
 }

 /*
 // detect minor piece draws
 if(((alpha > 0 && pos.wtm) || (beta < 0 && !pos.wtm)) 
    && !pos.plist[WHITE][PAWN][0] && pos.pieces[WHITE] < 4 && (!pos.plist[WHITE][BISHOP][0] || pos.pieces[WHITE] < 3) 
    && !pos.plist[WHITE][ROOK][0] && !pos.plist[WHITE][QUEEN][0]) return(0);
 if(((alpha > 0 && !pos.wtm) || (beta < 0 && pos.wtm)) 
    && !pos.plist[BLACK][PAWN][0] && pos.pieces[BLACK] < 4 && (!pos.plist[BLACK][BISHOP][0] || pos.pieces[BLACK] < 3) 
    && !pos.plist[BLACK][ROOK][0] && !pos.plist[BLACK][QUEEN][0]) return(0);
 */

 //-------------------
 //    Check EGTB
 //-------------------
 if(ply == 1 || (pos.last.b.type&CAPTURE)) {
   int total_pieces = pos.pieces[0]+pos.pieces[1]
     +pos.plist[0][PAWN][0]
     +pos.plist[1][PAWN][0];
   if(total_pieces == 2) {
     //return 0;
     if(ply > 1) return(MAX(MIN(0,beta),alpha));
     else return 0;
   }
#if TABLEBASES
   if((ply == 1 && total_pieces <= EGTB)
      || total_pieces <= MIN(EGTB,4)) {
     pthread_mutex_lock(&egtb_lock);
     score = probe_tb(&pos, ply);
     pthread_mutex_unlock(&egtb_lock);
     tdata->egtb_probes++;
     if(score != -1) {
       if(ply == 1) tdata->tb_hit = 1;
       tdata->egtb_hits++;
       //return(score);
       if(score == 0 && ply > 1) return(MAX(MIN(0,beta),alpha));
       else return(score);
     }
   }
#endif  /* TABLEBASES */
 }
 
 //------------------------------------------
 //    Check hash table
 //     --> adjust pos.hcode if in PV
 //     --> adjust for a skipped move, 
 //         if there is one..
 //------------------------------------------
 if(in_pv) Or(pos.hcode,h_pv);
 if(move_to_skip) Or(pos.hcode,move_to_skip);
 hscore = get_hash(&pos.hcode, &hflag, &hdepth, &pos.hmove, ply, &singular);
 // if(!move_to_skip) hscore = get_hash(&pos.hcode, &hflag, &hdepth, &pos.hmove, ply, &singular);
 // else hscore = HASH_MISS; 
 if(move_to_skip) Or(pos.hcode,move_to_skip);
 if(in_pv) Or(pos.hcode,h_pv);
       
 //---------------------------------
 // process the returned hash hit
 //---------------------------------
 if(hscore != HASH_MISS) {
   // see if we can return a score 
   if(hdepth >= depth) {    
     if(hflag == FLAG_P) {
       tdata->hash_count++;
       if(hscore > alpha) {
	 tdata->pc[ply][ply].t = pos.hmove.t;
	 tdata->pc[ply][ply+1].t = NOMOVE;
       }
       //return hscore;
       if(ABS(hscore) > MATE/2) return hscore;
       return MAX(MIN(hscore,beta),alpha);
     }
     if(hflag == FLAG_A && hscore <= alpha) {
       tdata->hash_count++;
       //return hscore;
       if(ABS(hscore) > MATE/2) return hscore;
       return MAX(MIN(hscore,beta),alpha);
     } 
     if(hflag == FLAG_B && hscore >= beta) {
       tdata->hash_count++;
       //return hscore;
       if(ABS(hscore) > MATE/2) return hscore;
       return MAX(MIN(hscore,beta),alpha);
     }
   }
   // set null-move switch
   if((hflag==FLAG_A || hflag==FLAG_P) && hscore < beta) null_hash = 0;
   // zero move if depth is too shallow
   if(hdepth < depth-3) { pos.hmove.t = NOMOVE; singular = 0; }
   // zero move if it cannot be verified
   //  --> should not be necessary as we have a 64-bit key and some
   //      checks in the exmove function to prevent illegal moves
   //if(pos.hmove.t) {
   //    if(!pos.verify_move(&moves,tdata,pos.hmove)) pos.hmove.t = NOMOVE;
   //}
 } 
   
 //---------------------------------------------
 // Generate a pre-move score for the position
 //---------------------------------------------
#if NNUE
 if(nnue_available && !acc.computed) nnue_apply_delta(acc, prev->acc, pos);
#endif
 if(pos.last.t) premove_score = pos.score_pos(gr,tdata NNUE_ACC_ARG);

 // ----------------------------------------------------------
 //     Null Move Heuristic
 // ----------------------------------------------------------
 if(!in_pv                               // Only outside of pv nodes
    && NULL_MOVE                         // Not switched off
    && pos.last.t                        // Don't repeat null or follow IID
    && premove_score > beta              // Must be a winning node
    && !pos.hmove.t                      // Don't have a hash move 
    && prev->pos.hmove.t != pos.last.t   // Not after an opponent's hash move
    && !pos.check                        // Not when in check
    && null_hash                         // Not if hash score too high
    && pos.pieces[pos.wtm] > 1           // Not if too few pieces
    && beta > -(MATE/2)                  // Not when getting mated
    && ply < MAXD-3) {                   // Not near end of allowed search 
   //----------------------
   // Regular Null Move
   //----------------------
     //----------------------
     // make the null move
     //----------------------
     next->pos = pos; next->pos.wtm ^= 1;
     Or(next->pos.hcode,hstm);
     if(pos.ep) Or(next->pos.hcode,ep_code[FILE(pos.ep)]);
     next->pos.last.t = NOMOVE; next->pos.ep = 0;
     next->pos.material = -next->pos.material;
     next->premove_score = -premove_score;
     next->pos.fifty = 0;
#if NNUE
     if(nnue_available) { next->acc = acc; next->acc.computed = true; }
#endif
     // clear old downstream checks so they don't trigger threat check
     next->next->pos.check = 0;
     //------------------------------------------------
     // set reduction depth depending on premove score
     //------------------------------------------------
     int R = 3;
     if(premove_score > beta+400 && depth >= 6) { R = 5; }
     else if(premove_score > beta+100 && depth >= 5) { R = 4; }
     //----------------------
     // do the search
     //----------------------
     if(depth >= R+1) {
       score = -next->pvs(-beta, -beta+1, depth-R-1, 0, 0);
     } else {
       score = -next->qsearch(-beta, -beta+1, 0);
     }
     if(score == TIME_FLAG) return -TIME_FLAG;
     //----------------------
     // return a cutoff
     //----------------------
     if (score >= beta) {
       tdata->null_cutoff++;
       //put_hash(&pos.hcode, score, save_alpha, beta, depth, pos.hmove.t, ts->h_id, ply);
       return beta;
     }
     // Identify a threat square
     if(next->pos.hmove.b.type&CAPTURE) {
       pos.threat_square = next->pos.hmove.b.to;
     }
     // Identify a check threat square
     if(next->next->pos.check && pos.qchecks[pos.wtm^1] > 0) {
       pos.threat_check = 1;
     }
 } 

 // --------------------------------------------------------------------
 //     Assign reply, combination, and pawn hash moves
 //     to be given higher scores in movelist
 //     -- The 'reply' idea is the 'counter-move' heuristic
 //        suggested by by Jos Uiterwijk, 
 //        (see http://chessprogramming.wikispaces.com/Countermove+Heuristic)
 //     -- The combination hash move is apparently the same in
 //        functionality as Variant 1 of Steven Edwards LBR idea
 //        (see http://chessprogramming.wikispaces.com/Last+Best+Reply)
 //        but the idea here is to use the hash code of the current 
 //        position and the hash code of 2 ply earlier to identify the
 //        unique hash signature of the moves made in between, see
 //        comments below for restriction on what moves can be added
 //        to this table when they fail high.
 // --------------------------------------------------------------------
 if(pos.last.t || prev->pos.wtm != pos.wtm) 
   {  pos.rmove.t = 0; pos.cmove.t = 0; }

 ckey =0ULL;  // *must* set this to prevent cmove from being undefined
              //   and probing an area of memory that is not allowed
 
 if(pos.last.t) { 
   // counter (reply) move  format: reply[PIECE][TO-SQUARE]
   pos.rmove.t = tdata->reply[pos.sq[pos.last.b.to]][pos.last.b.to]; 
   // combination hash move
   if(prev->pos.last.t && ts->turn+ply-3 >= 0) {
     ckey = ((tdata->plist[ts->turn+ply-3]^pos.hcode));
     if(pos.wtm) { ckey ^= hstm; }
     cmove = cmove_table + (((CMOVE_SIZE-1)*((ckey)&MAX_UINT))/MAX_UINT);
     //cmove = cmove_table + (ckey&(CMOVE_SIZE-1));
     // use depth priority 2-stage table, probe the first layer first and
     //  then the second layer.  Note: I tried a condition where if the
     //  combination move == hmove, then go to the second layer of the table
     //  to find another cmove, but this failed terribly... It must be because
     //  if the stored cmove == hmove in the top layer, then the hash move
     //  IS the correct combination move and trying to take a different lower
     //  quality move from the second layer will hurt not help.  Note setting
     //  cmove = hmove will have no effect as hmove will still be scored first, so
     //  it is harmless to allow this
     if(cmove->depth1 >= depth-3) { 
       pos.cmove.t = cmove->move1;
       // check for a key match after data assignment (to be sure it doesn't change
       //  due to other threads working in the table)
       if(ckey != cmove->get_key1()) pos.cmove.t = NOMOVE;  
     } 
     if(!pos.cmove.t && cmove->depth2 >= depth-3) {
       pos.cmove.t = cmove->move2;
       // check for a key match after data assignment (to be sure it doesn't change
       //  due to other threads working in the table)
       if(ckey != cmove->get_key2()) pos.cmove.t = NOMOVE;  
     }
   }
 }

 // --------------------------------------------------------
 //     Internal Iterative Deepening (IID) if no hash move
 //     -- idea taken from crafty
 //        modified by Ed Schroeder's idea to do this even 
 //        outside the main pv line -> seems to work well
 //     -- not when we are skipping a move (for sing. ext)
 //     -- not *after* a hmove in the prev node, unless in PV
 // --------------------------------------------------------
 if(pos.hmove.t) { tdata->hmove_count++; }
 else if(depth > 3
 	 && !move_to_skip 
    	 && (in_pv || prev->pv_node || prev->pos.hmove.t != pos.last.t || !pos.last.t)) {
    tdata->internal_iter++;
    next->pos = pos; next->pos.last.t = NOMOVE;
    next->premove_score = premove_score;
    depth_mod = prev->depth_mod;
#if NNUE
    if(nnue_available) { next->acc = acc; next->acc.computed = true; }
#endif
#if DEBUG_SEARCH
    search_outfile << space_string << "Entering an IID loop\n";
#endif
    hscore = next->pvs(alpha, beta, depth-3, in_pv, 0);
#if DEBUG_SEARCH
    search_outfile << space_string << "Leaving an IID loop\n";
#endif
    if(hscore == -TIME_FLAG) { return -TIME_FLAG; }
    if(hscore > alpha) { 
      // Assign best move to try
      pos.hmove.t = next->pos.hmove.t;
      // Prevent singular searches on this move
      hflag = 0; 
      if(hscore >= beta && beta > alpha+1) { hflag = FLAG_B; } 
      //premove_score = MAX(premove_score,hscore);
    }
 } 

 // --------------------------------------------------------
 //     Generate Move List
 //     -- the following function call also scores 
 //        moves so that they can be sorted during
 //        the move loop below
 // --------------------------------------------------------
 int generated_moves = 0;
 if(!pos.hmove.t || pos.check) { 
   pos.allmoves(&moves, tdata);
   assert(moves.count < MAX_MOVES);
   generated_moves = 1;
 } else { 
   moves.count = 2;
 }

 // --------------------------------------
 //   Singular extension test
 // --------------------------------------

 #define SMARGIN  25
 #define EASY_SMARGIN  75

 if(depth > 8
    && (in_pv || prev->pv_node || prev->pos.hmove.t == pos.last.t || !(pos.hmove.b.type&CAPTURE))
    && pos.hmove.t
    && pos.last.t
    && (!singular || ply == 2)
    && pos.pieces[pos.wtm] > 1
    && (hscore > alpha && alpha > -MATE/2)
    && hflag > FLAG_A 
    //&& moves.count > 1
    && ply + depth < MAXD-3) {

    next->pos = pos; next->pos.last.t = NOMOVE;
    next->premove_score = premove_score;
    depth_mod = prev->depth_mod;
#if NNUE
    if(nnue_available) { next->acc = acc; next->acc.computed = true; }
#endif
#if DEBUG_SEARCH
    search_outfile << space_string << "Entering an Singular Ext. loop\n";
#endif
    int test_score = hscore-SMARGIN;
    if(ply == 2) test_score = hscore-EASY_SMARGIN;  // larger margin if this might result in an easy move
    int test_depth = depth/2;
    score = next->pvs(test_score-1,test_score, test_depth, in_pv, pos.hmove.t);
    if(score == -TIME_FLAG) return -TIME_FLAG;
#if DEBUG_SEARCH
    search_outfile << space_string << "Leaving an Singular Ext. loop\n";
#endif
    // if score < test_score -- the first move is singular
    singular = 0;
    if(score < test_score) {
      singular = 1;
    }
    /*
    else if(next->pos.hmove.t) {
      if(!pos.cmove.t || pos.cmove.t == pos.hmove.t) pos.cmove.t = next->pos.hmove.t;
      else if(!pos.rmove.t || pos.rmove.t == pos.hmove.t) pos.rmove.t = next->pos.hmove.t;
      else if(pos.rmove.t == pos.cmove.t) pos.rmove.t = next->pos.hmove.t;
    }
    */
 } // closing bracket of singular search code


 // --------------------------------------------------------
 //     Move Loop
 //     Basic plan is as follows...
 //     -- loop over each move from the list
 //     -- make the move to the position in the "next" node
 //     -- setup any extensions
 //     -- do any pruning or depth reductions
 //     -- search the "next" node to depth-UNIT_DEPTH
 //     -- use score from move and alpha/beta cuts
 //        to decide if we are done (fail high) or
 //        need to continue... do associated book
 //        keeping
 // --------------------------------------------------------
 mcount = 0; first = 1; skipped = 0;
 while (mcount < moves.count+skipped && best < beta)
 {

   //------------------------------------------
   //    Check hash table
   //     --> if a move has been skipped
   //         as another thread is searching
   //         and may produce a cutoff
   //     --> suggested by Tom Kerrigan 
   //         on CCC
   //------------------------------------------
   if(skipped && depth > 1) {
     if(in_pv) Or(pos.hcode,h_pv);
     if(move_to_skip) Or(pos.hcode,move_to_skip);
     int tscore, tflag, tdepth, tsingular; move tmove;
     tscore = get_hash(&pos.hcode, &tflag, &tdepth, &tmove, ply, &tsingular);
     if(move_to_skip) Or(pos.hcode,move_to_skip);
     if(in_pv) Or(pos.hcode,h_pv);
       
     //---------------------------------
     // process the returned hash hit
     //---------------------------------
     if(tscore != HASH_MISS) {
       // see if we can return a score 
       if(tdepth >= depth) {    
	 if(tflag == FLAG_P) {
	   tdata->hash_count++;
	   if(tscore > alpha) {
	     tdata->pc[ply][ply].t = pos.hmove.t;
	     tdata->pc[ply][ply+1].t = NOMOVE;
	   }
	   return tscore;
	 }
	 if(tflag == FLAG_A && tscore <= alpha) {
	   tdata->hash_count++;
	   return tscore;
	 } 
	 if(tflag == FLAG_B && tscore >= beta) {
	   tdata->hash_count++;
	   return tscore;
	 }
       }
     }
   }

   //--------------------------------------------
   // Generate all moves after the first one if
   // we have not already done so
   //--------------------------------------------
   if(mcount == 1 && !generated_moves) {
     pos.allmoves(&moves, tdata);
     assert(moves.count < MAX_MOVES);
     // Sort the first move so we don't search it again
     MSort(&moves.mv[0], &moves.mv[moves.count-1]);
     generated_moves = 1;
     if(moves.count == 1) break; 
   }

   // assign the move index
   if(mcount >= moves.count) {
     mi = skip_move[mcount-moves.count];
   } else {
     mi = mcount;
   }

   if(generated_moves) {
     // ----------------------------------------------------------------
     // shuffle highest scored move to top of list
     // ----------------------------------------------------------------
     if(mcount < moves.count-1) { 
       MSort(&moves.mv[mi], &moves.mv[moves.count-1]);
     }
     smove = moves.mv[mi].m;
   } else {
     smove = pos.hmove;
     moves.mv[0].score = 50000000;
     moves.mv[0].m = smove;
   }

   // Skip move specified
   if(smove.t == move_to_skip) {
     mcount++;
     continue;
   }

   //-----------------------------------------------
   // Loop over other the share data for other 
   //  threads to see if this move is being
   //  searched by another and should be skipped
   //  for the time being.  The move will be 
   //  added to a list to be visited at the end
   //  of the movelist.
   // -- Only helps in pv nodes
   //-----------------------------------------------
   if(in_pv && THREADS > 1 && !first && skipped < 23 && mcount < moves.count) {
     int skip = 0;
     for(int ti = 0; ti < THREADS; ti++) {
       if(ti == tdata->ID) continue;
       if(ts->share_data[ply][ti].check_active_move(pos.hcode,depth,save_alpha,beta,smove.t)) {
	 skip = 1;
	 break;
       }
     }
     if(skip) {
       skip_move[skipped] = mi;
       skipped++;
       mcount++;
       continue;
     }
   }
 
   // -------------------------------------------------------------
   // Allow possibility for an oversight when knowledge_scale < 50
   // -------------------------------------------------------------
   if(gr->knowledge_scale < 50 && !first) {
     if((((pos.hcode^(smove.b.to+smove.b.from))&63ULL) > gr->knowledge_scale)
	&& (((pos.hcode^smove.b.to^smove.b.from)&3)==3)) {
       mcount++;
       continue;
     }
   }

   // -------------------------------------------------
   // making move in the next position in the search
   //   -- if it is illegal, skip to next move in list
   // -------------------------------------------------
   next->pos = pos;
   if(!next->pos.exec_move(smove, ply)) {
    mcount++;
    continue;
   }
#if NNUE
   if(nnue_available) nnue_record_delta(next->acc, pos, next->pos, smove);
#endif

   //---------------------------------------------
   // now set this move as the one being worked
   //---------------------------------------------
   if(in_pv && THREADS > 1) {
     ts->share_data[ply][tdata->ID].set_active_move(pos.hcode,depth,save_alpha,beta,smove.t);
   }

   // Prefetch the hash line for this position so it can be accessed quickly in next node
   __builtin_prefetch((hash_bucket *)(hash_table + (((TAB_SIZE-1)*((next->pos.hcode)&MAX_UINT))/MAX_UINT)));
   __builtin_prefetch((score_rec *)(score_table + (((SCORE_SIZE-1)*((next->pos.hcode)&MAX_UINT))/MAX_UINT))); 
   /*   
   if(pos.last.t && ts->turn+ply-2 >= 0) {
     __builtin_prefetch((cmove_rec *)(cmove_table + (((CMOVE_SIZE-1)*(((tdata->plist[ts->turn+ply-2]^next->pos.hcode))&MAX_UINT))/MAX_UINT)));
   }
   */

#if DEBUG_SEARCH
   pos.print_move(smove, last_move_string, tdata);
   search_outfile << space_string << "Move: " << last_move_string;
   search_outfile << " Score: " << moves.mv[mi].score << "\n"; 
#endif
 
   // -------------------------------------------------------
   // initalize depth modifications for this move to be zero    
   // -------------------------------------------------------
   depth_mod = 0; 

   // -----------------------------------
   //   Set the extensions
   // -----------------------------------
   if(depth+ply < (MAXD-3)) {   
     //-------------------------------------------------------
     // If last move was the only available move from a check
     //-------------------------------------------------------
     if(moves.count == 1 && pos.check)
       { depth_mod = 1; tdata->extensions++; }
     //--------------------
     // It is singular 
     //--------------------
     else if(mi == 0 && singular)
       { depth_mod = 1; tdata->extensions++; tdata->sing_count++; }
     //-------------------------------------------------------------------------
     // Check extension for pv and near pv nodes or cases of king vulnerability
     //-------------------------------------------------------------------------
     else if(next->pos.check && (in_pv || prev->pv_node || pos.qchecks[pos.wtm] > 0)  
	     && moves.mv[mi].score > 1000000) 
       { depth_mod = 1; tdata->extensions++; }
   }

#if DEBUG_REDUCTION
   int attempted = 0;
#endif

   // see if this move is an effective evasion attempt of the threat: 
   //   if not, do normal pruning and reducing
   if(pos.threat_square < 64 && smove.b.from == pos.threat_square
      && ((!(smove.b.type&CAPTURE) && swap(smove.b.to,pos,pos.wtm,smove.b.from) >= 0)
	  || ((smove.b.type&CAPTURE) && moves.mv[mi].score > 9900000))) {
     // reset threat square after we've made a reasonable
     //  evasion attempt... if we can't score > alpha by avoiding
     //  this threat, then further attempts to do so are also
     //  likely to be futile
	 pos.threat_square = 65;
   } else if(pos.threat_check && PTYPE(pos.sq[smove.b.from]) == KING) {
     //threat_check = 0;
   }
   // ---------------------------------------------------
   //   Move Reductions and Pruning
   // ---------------------------------------------------
   else if(!depth_mod && !first && !pos.check && !next->pos.check 
	   && (!in_pv || pos.hmove.t != tdata->pc[0][ply].t || best >= alpha)
	   && (premove_score < alpha+mcount*mcount*200)
	   && best > -(MATE/2)) {                                // best > -(MATE/2) important!

     //----------------------------------------------------------------------
     // First see if we can abort the search or reduce the move
     //   Only moves scored as less than ... 
     //----------------------------------------------------------------------
     if(moves.mv[mi].score < 2000000) {       	      // not a killer, safe check, or safe pawn push to 7th
       // -----------------------------------------------------------------
       // Search Abort Test
       //   -- if we are deep in the movelist in a zero-window search, 
       //      this is likely an ALL node, so we can just stop searching 
       //      moves now as none are expected to be > alpha.
       //   -- similar idea to late move pruning, which may be more
       //      powerful as it tests and prunes moves individually. 
       //      Need to try that as an alternative at some point
       // -----------------------------------------------------------------
       if(depth < 7 && skipped == 0 && !move_to_skip) {        
	 int index = depth+(premove_score-beta)/100;
	 if(index < 0) index = 0;
	 else if(index > 7) index = 7;
	 if(mcount > (abort_search_fraction[index]*moves.count)/1000) {
	   // if there is no identified threat square, just terminate the search
	   if(pos.threat_square >= 64) {
	     //tdata->node_count++;
	     //SEARCH_INTERRUPT_CHECK();
	     //put_hash(&pos.hcode, alpha, save_alpha, beta, depth, pos.hmove.t, ts->h_id, ply);
	     return alpha;
	   } else {
	     // otherwise just prune this move until we've come to
	     // a move that allows a reasonable evasion attempt, see above
	     //tdata->node_count++;
	     //SEARCH_INTERRUPT_CHECK();
	     mcount++;
	     continue; 	   
	   }
	 }
       }
       if(depth >= 1) {
	 // -----------------------------------
	 // try move reduction 
	 // -----------------------------------
	 depth_mod -= 1;
	 if(depth < 5) depth_mod -= lmr_mi8[mi];
	 else {
	   if(premove_score < alpha-350) depth_mod -= lmr_d5[depth];
	   //------------------------------------------------------------------
	   // make another reduction decision based on history or position in move list
	   // --> reduce if this move loses historically (~ 90% fail low history)
	   //------------------------------------------------------------------
	   if(moves.mv[mi].score < -10*depth) {
	     depth_mod -= 1 + lmr_d12[depth];
	     if(moves.mv[mi].score < -23*depth) {
	       depth_mod -= 1 + lmr_d8[depth];
	     }
	   }
	 }
       }
     }

     //----------------------------------------------------------------------
     // Futility Pruning for moves that cannot get us near alpha but don't 
     //  trigger a search abort above
     //----------------------------------------------------------------------     
     int pawn_bonus = 0;
     if(PTYPE(pos.sq[smove.b.to]) == PAWN 
	&& (RANK(smove.b.to) == 6 || RANK(smove.b.to) == 1)) {
       pawn_bonus = 35*pos.gstage;
     } 
     if(!in_pv && !prev->pv_node &&
	depth+depth_mod < 4 && premove_score+MARGIN(MAX(0,(depth+depth_mod)))
	+pawn_bonus+value[PTYPE(pos.sq[smove.b.to])]
	+value[PTYPE(smove.b.promote)] < alpha) {
       //tdata->node_count++;
       //SEARCH_INTERRUPT_CHECK();
       mcount++;
       continue;
     }

   }

   // -----------------------------------
   //   Search the next node in the tree
   // -----------------------------------
   if(depth+depth_mod < 1) {
     score = -next->qsearch(-beta, -alpha, 0);
     if (score == TIME_FLAG) return -TIME_FLAG;
     // Checking move reductions with a re-search    
     if (depth && score > alpha && score < (MATE/2)) { 
	depth_mod = 0; 
	score = -next->pvs(-beta, -alpha, depth+depth_mod-1, in_pv, 0);
	if (score == TIME_FLAG) return -TIME_FLAG;
     }
   } else {
    if(first) {
      score = -next->pvs(-beta, -alpha, depth+depth_mod-1, in_pv, 0);
      if (score == TIME_FLAG) return -TIME_FLAG;    
    } else {
      score = -next->pvs(-alpha-1, -alpha, depth+depth_mod-1, 0, 0);
      if (score == TIME_FLAG) return -TIME_FLAG;
      if (score > alpha && (depth_mod < 0 || beta > alpha+1) && score < (MATE/2)) { 
        // Now research
	if(depth_mod < 0) depth_mod = 0;  // for checking move reductions on re-search
	score = -next->pvs(-beta, -alpha, depth+depth_mod-1, in_pv, 0);
	if (score == TIME_FLAG) return -TIME_FLAG;
      }
    }
   }
  
#if DEBUG_SEARCH
 search_outfile << space_string << "Returned value: " << score << "\n";
#endif

#if DEBUG_REDUCTION
 if(attempted && score <= alpha) {
   successes++;
 }
#endif

   // ------------------------------------ 
   //   Update best score
   // ------------------------------------ 
   if(score > best) { 
     best = score;
   }

   // ------------------------------------ 
   //   If score is not a fail-low
   //     -- increase alpha bound
   //     -- store best move
   //     -- update pv
   //     -- update history/killer moves
   //         if this is a fail high
   //   Else
   //     -- reduce history move score
   // ------------------------------------ 
   if(score > alpha) {    
     alpha = score;
     pos.hmove = smove;
     // invalidate singular status
     if(mi!=0) singular = 0;
     //------------------------------------------------------------------------
     // if score >= beta, increase fail high counts and update history/killers
     //------------------------------------------------------------------------
     if(score >= beta) {
       tdata->fail_high++;
       if(!mcount) tdata->first_fail_high++;
       if(!move_to_skip && (pos.last.t || prev->pos.wtm == pos.wtm) && beta > -(MATE/2)) {
	 //-----------------------------------------------------------------------
	 // update combination move hash for scoring
	 //  other nodes with the same combination, see above for description
	 //  --  don't include forced responses to checks
 	 //  --  require that the move NOT be a winning capture as these will
 	 //      be ranked highly in other nodes anyway 
	 //  --  use a 2-layer, depth priority table to help ensure quality
	 //      of the moves stored
	 //-----------------------------------------------------------------------
	 if(ckey && !pos.check 
	   // && !((smove.b.type&CAPTURE) && moves.mv[mi].score >= 10000000)) { 
	    && !((smove.b.type&CAPTURE) && value[PTYPE(pos.sq[smove.b.to])] >= value[PTYPE(pos.sq[smove.b.from])])) { 
	   if(cmove->id != ts->h_id || depth > cmove->depth1) {
	     if(ckey != cmove->get_key1() || smove.t != cmove->move1) {
	       cmove->move2 = cmove->move1;
	       cmove->depth2 = cmove->depth1;
	       cmove->key2 = cmove->key1;
	     }
	     cmove->move1 = smove.t;         // store move in first layer
	     cmove->id = ts->h_id;
	     cmove->depth1 = depth;
	     cmove->set_key1(ckey,smove.t,depth);
	   } else {                          // otherwise store in second layer
	     cmove->move2 = smove.t;
	     cmove->depth2 = depth;
	     cmove->set_key2(ckey,smove.t,depth);
	   }
	 }
	 //--------------------------------------------
	 // update history list, reply list, killer moves
	 //   -- reply idea is 'counter-move' killer heuristic
	 //      introduced by Jos Uiterwijk, see http://chessprogramming.wikispaces.com/Countermove+Heuristic
	 //   -- only update history score when
	 //      we are more than two ply from leaf
	 //--------------------------------------------
	 if(!(smove.b.type&CAPTURE)) {
	   // update reply (countermove) list and pawn hash move
	   if(pos.last.t && moves.mv[mi].score < 8000000) {
	     // Format:  format: reply[PIECE][TO-SQUARE]
	     tdata->reply[pos.sq[pos.last.b.to]][pos.last.b.to] = smove.t;
	   }
	   // update history score for this move 
	   // -- NOT when in check OR if this is a checking move
	   if(depth+depth_mod > 1 && !pos.check && !next->pos.check 
	      && tdata->history[pos.sq[smove.b.from]][smove.b.to] < 1000000) {
	     tdata->history[pos.sq[smove.b.from]][smove.b.to] += 11*(depth+depth_mod);
	   }
	   // update killers -- NOT when in check
	   if(!pos.check) {
	     if(tdata->killer1[pos.wtm] != smove.t) {
	       tdata->killer3[pos.wtm] = tdata->killer2[pos.wtm];
	       tdata->killer2[pos.wtm] = tdata->killer1[pos.wtm];
	       tdata->killer1[pos.wtm] = smove.t;
	     }
	   }
	 }
       }
     } else { 
       tdata->pc_update(smove, ply);  // update principle continuation if this is an exact score
       // set singular status in overall principle continuation
       // --> for possible 'easy' move detection
       if(ply == 2 && pos.last.t && singular) {
	 tdata->pc[2][2].b.type |= SINGULAR;
       }
     }

   } else {   /*  following is for when we fail low */

     //--------------------------------------------------------
     // If move is first or second and we don't have a threat square
     // try to identify one...
     //--------------------------------------------------------
     if(pos.threat_square == 64 
        && mcount < 3
	&& premove_score > beta
	&& (next->pos.hmove.b.type&CAPTURE)
	&& next->pos.hmove.b.to != smove.b.to) {
	 pos.threat_square = next->pos.hmove.b.to;
     }

     //--------------------------------------------------------
     // adjust score in history table for failing low
     //   -- NOT when in check OR if this is a checking move
     //--------------------------------------------------------
     if(depth+depth_mod > 1 && !pos.check && !next->pos.check && !(smove.b.type&CAPTURE) 
	&& tdata->history[pos.sq[smove.b.from]][smove.b.to] > -1000000) 
       tdata->history[pos.sq[smove.b.from]][smove.b.to] -= (depth+depth_mod); 
 
     // If move is first, then it is not singular
     if(mi == 0) singular = 0;

   }  

   // ------------------------------------ 
   // update move loop variables
   // ------------------------------------    
   first = 0;
   mcount++;
 
 }  /* End of move loop */     

 //---------------------------------------------
 // Clear active move data for shared work
 //---------------------------------------------
 if(in_pv && THREADS > 1) { 
   ts->share_data[ply][tdata->ID].set_active_move(0,0,0,0,0);
 }

 // ----------------------------------------------------------------- 
 // if it is mate or stalemate; no move is in principle continuation
 // ----------------------------------------------------------------- 
 if(first && mcount == moves.count && !move_to_skip) {
   if(!pos.check) best = 0;        // score is even if stalemate
   pos.hmove.t = NOMOVE;
 } 

 // ------------------------------------------------------ 
 // storing position in the hash table
 //  -- assign the hash move to be the best move if the
 //     position is a fail-low with no best move
 //  -- Do a double store for the PV
 //  -- Adjust hash signature for a skipped move, if there is one
 // ------------------------------------------------------ 
 if((best != 0 || save_alpha == 0)) {   // don't store draws depending on path
 // if(!move_to_skip && (best != 0 || save_alpha == 0)) {   // don't store draws depending on path
   // store singular status in hash move structure itself
   //  --> so it will be stored in table and can be extracted
   if(singular) pos.hmove.b.type |= SINGULAR;
   if(move_to_skip) Or(pos.hcode,move_to_skip);
   put_hash(&pos.hcode, best, save_alpha, beta, depth, pos.hmove.t, ts->h_id, ply); 
   if(in_pv) { 
     Or(pos.hcode,h_pv);
     put_hash(&pos.hcode, best, save_alpha, beta, depth, pos.hmove.t, ts->h_id, ply); 
     Or(pos.hcode,h_pv);
   }
   if(move_to_skip) Or(pos.hcode,move_to_skip);
   // erase singular status in hash move structure 
   if(singular) pos.hmove.b.type &= NOT_SINGULAR;
 }

 //return best;
 if(best < -(MATE/2) && best < alpha) return best;
 if(ABS(best) > MATE/2) return best;
 return MAX(MIN(best,beta),alpha);
 //return best;

}

/*--------------------- Quiescent search ---------------------*/
// This searches only non-losing captures.  Futility cut-offs
// are made if the capture is not likely to bring us up back
// above alpha.   A straight alpha-beta algorithm is used here.
int search_node::qsearch(int alpha, int beta, int qply)
{
  int qi, delta_score; 
 
  // save the original alpha score for node
  save_alpha = alpha;
 
  // increase node counts
  tdata->node_count++; tdata->q_count++;


  // ----------------------------------------------------------
  //     Check search interrupt... defined in search.h
  // ----------------------------------------------------------
  SEARCH_INTERRUPT_CHECK();

  //--------------------------------------------------
  //    Add hash code for this position to position list
  //--------------------------------------------------
  tdata->plist[ts->turn+ply-1] = pos.hcode;
  
  //--------------------------------------------------
  //    Do Draw Testing on first layer of Q-search
  //--------------------------------------------------
  if(pos.last.t && !qply) {	   
    // fifty move rule
    if(pos.fifty >= 100) {
      //return 0;
      return(MAX(MIN(0,beta),alpha));
    }
   // avoid repeating a position if possible
   int test_ply = ply;
   for(int ri = ts->turn+test_ply-3; ri >= ts->turn+test_ply-pos.fifty-1; ri -= 2) {
     // account for IID or singular tests in move sequence
     while(ri > 0 && tdata->plist[ri] == tdata->plist[ri-1]) { ri--; test_ply--; }
     // check code for rep.
     if(tdata->plist[ri] == pos.hcode) {
       //return 0;
       return(MAX(MIN(0,beta),alpha));
     }
   }
  }

#if DEBUG_SEARCH
  // Debug archiving of search to a file
  char space_string[50] = "\0";
  char last_move_string[10];
  for(int si = 0; si < ply; si++) strcat(space_string," ");
  if(ply) prev->pos.print_move(pos.last, last_move_string, tdata);
  
  search_outfile << space_string << "Qsearch->Ply: (ply, qply) (" << ply << ", " << qply <<"), max_ply: "
	  << ts->max_ply << ", alpha: " << alpha
	  << ", beta: " << beta << ", last: " << last_move_string << "\n";
#endif

  //-------------------
  //    Check EGTB
  //-------------------
  if(ply == 1 || (pos.last.b.type&CAPTURE)) {
    int total_pieces = pos.pieces[0]+pos.pieces[1]
      +pos.plist[0][PAWN][0]
      +pos.plist[1][PAWN][0];
    if(total_pieces == 2) {
      if(ply > 1) return(MAX(MIN(0,beta),alpha));
      else return 0;
      //return 0;
    }
#if TABLEBASES
    if((ply == 1 && total_pieces <= EGTB)
       || total_pieces <= MIN(EGTB,4)) {
      pthread_mutex_lock(&egtb_lock);
      score = probe_tb(&pos, ply);
      pthread_mutex_unlock(&egtb_lock);
      tdata->egtb_probes++;
      if(score != -1) {
	if(ply == 1) tdata->tb_hit = 1;
	tdata->egtb_hits++;
	if(score == 0 && ply > 1) return(MAX(MIN(0,beta),alpha));
	else return(score);
	//return(score);
      }
    }
#endif  /* TABLEBASES */
  }
 
  //--------------------------------------------------------
  // must put a blank move in the principle continuation
  // because we may choose to make no move during qsearch
  //--------------------------------------------------------
  tdata->pc[ply][ply].t = NOMOVE;
  pos.hmove.t = NOMOVE;
  hflag = 0;
  pos.threat_square = 64;
  pv_node = 0;
  singular = 0;
 
  //--------------------------------------------------------
  // break the search if we are at the end of allowed plys
  //  -- could also consider this if we are in check
  //     after the first qsearch ply, with the idea that
  //     the opponent is guaranteed at least this score...
  //     Seems to work about as well as searching all check
  //     evasions, but not better, so not doing this now
  //--------------------------------------------------------
#if NNUE
  if(nnue_available && !acc.computed) nnue_apply_delta(acc, prev->acc, pos);
#endif
  if(ply >= MAXD-2 || (pos.check && qply > 0))
    return MAX(MIN(pos.score_pos(gr, tdata NNUE_ACC_ARG),beta),alpha);

  //---------------------------  
  // On first layer of q-search 
  // - reset qsearch nodes
  // - check hash table 
  //---------------------------  
  //if(!qply) {
    hscore = get_hash(&pos.hcode, &hflag, &hdepth, &pos.hmove, ply, &singular);
    if(hscore != HASH_MISS) {
      // see if we can return a score
      if(hdepth >= -1) { 
	if(hflag == FLAG_P) {
	  tdata->hash_count++;
	  if(hscore > alpha) {
	    tdata->pc[ply][ply].t = pos.hmove.t;
	    tdata->pc[ply][ply+1].t = NOMOVE;
	  }
	  //return(hscore);
	  if(ABS(hscore) > MATE/2) return hscore;
	  return MAX(MIN(hscore,beta),alpha);
	}
	if(hflag == FLAG_A && hscore <= alpha) {
	  tdata->hash_count++;
	  //return(hscore);
	  if(ABS(hscore) > MATE/2) return hscore;
	  return MAX(MIN(hscore,beta),alpha);
	} 
	if(hflag == FLAG_B && hscore >= beta) {
	  tdata->hash_count++;
          //return(hscore);
	  if(ABS(hscore) > MATE/2) return hscore;
	  return MAX(MIN(hscore,beta),alpha);
	}
      }
    }
    //}

  //-------------------------------------
  // Now the rest of the normal qsearch
  //-------------------------------------
  delta_score = -50;   // set default futility cutoff

  // ------------------------------------ 
  //  Generate a score for the position
  // ------------------------------------ 
  if(!pos.check) { 
    if(pos.last.t) {  // we already have a score if the last was a null move
      // Generate a pre-move score for the position
      premove_score = pos.score_pos(gr, tdata NNUE_ACC_ARG);
    }
    best = premove_score;
    if(best < alpha && alpha < (MATE/2)) { 
      // set futility cutoff 'delta_score' which is used
      // in the generation of captures -- modified during the
      // capture generation depending on the type of piece
      // captured
      delta_score = alpha - best - 50;     
      best = alpha;
    }
    else if(best >= beta) {               // return (stand pat) if we already have a high enough score
      put_hash(&pos.hcode, best, save_alpha, beta, -1, NOMOVE, ts->h_id, ply);
      //return best;
      return MAX(alpha,MIN(best,beta));
    }
  }
  else { best = -MATE+ply; } 

  if(best > alpha && !pos.check) alpha = best;    // increase alpha if best > alpha

  // --------------------------------------------------------------- 
  // generate captures 
  //   + checks if eval says king in danger, but not if the situation 
  //     is hopeless (alpha - best >= some value).  OR if this is the
  //     the continuation of a forced check sequence.
  // OR generate full move list to evade a check 
  // --------------------------------------------------------------- 
  if(!pos.check && alpha-best < NO_QCHECKS  
        && ((!qply && pos.qchecks[pos.wtm] > 0) 
    	 || (qply <= 2 && prev->pos.check && prev->moves.count == 1))) 
    { tdata->qchecks++; pos.captchecks(&moves, delta_score); } 
  /*
  if(!pos.check && alpha-best < NO_QCHECKS 
     && qply <= 2 && prev->pos.check && prev->moves.count == 1) 
    { tdata->qchecks++; pos.captchecks(&moves, delta_score);} 
  */
  else if(!pos.check) pos.captures(&moves, delta_score);
  else { 
    pos.cmove.t = 0; pos.rmove.t = 0; 
    pos.allmoves(&moves,tdata);
  }

  assert(moves.count < MAX_MOVES);
 
  // ------------------------------------------------------
  // loop trough possible captures, trying them in turn
  // ------------------------------------------------------
  for (qi = 0; qi < moves.count && best < beta; qi++)
  {
    // shuffle highest scored move to top of list
    MSort(&moves.mv[qi], &moves.mv[moves.count-1]);

    smove = moves.mv[qi].m;

    // execute move
    //   -- returns a zero if the move leaves us in check
    next->pos = pos;
    if(!next->pos.exec_move(smove, ply)) { continue; }
#if NNUE
    if(nnue_available) nnue_record_delta(next->acc, pos, next->pos, smove);
#endif

    // Prefetch the hash line for this position so it can be accessed quickly in next node
    __builtin_prefetch((hash_bucket *)(hash_table + (((TAB_SIZE-1)*((next->pos.hcode)&MAX_UINT))/MAX_UINT)));
    //__builtin_prefetch((score_rec *)(score_table + (((SCORE_SIZE-1)*((next->pos.hcode)&MAX_UINT))/MAX_UINT)));

    // call next iteration to obtain a score
    score = -next->qsearch(-beta, -alpha, qply+1);
    if (score == TIME_FLAG) return -TIME_FLAG; 

    if(score > alpha) {
      tdata->pc_update(smove, ply);     
      alpha = score;
    }  

    if(score > best) { 
      best = score;     
      pos.hmove.t = smove.t;
    }
    
  }
  
  // --------------------------------------------- 
  // update hash 
  // --------------------------------------------- 
  //if(!qply) put_hash(&pos.hcode, best, save_alpha, beta, -1, pos.hmove.t, ts->h_id, ply);
  put_hash(&pos.hcode, best, save_alpha, beta, -1, pos.hmove.t, ts->h_id, ply);

  // return the best score
  if(best < -(MATE/2) && best < alpha) return best;
  if(ABS(best) > MATE/2) return best;
  return MAX(alpha,MIN(best,beta));
  //return best;

}






