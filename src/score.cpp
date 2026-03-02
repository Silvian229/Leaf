// EXchess source code, (c) Daniel C. Homan  1997-2017
// Released under the GNU public license, see file license.txt

#include "define.h"
#include "chess.h"
#include "funct.h"
#include "score.h"
#include "hash.h"
#include "extern.h"
#include <cmath>
#include <iostream>

#define KING_SAFETY_STAGE 10
#define BITMASK(x) (1<<(x))
#define BITMASK_64(x) (1ULL<<(x))

//-------------------------------------------
//
// Inline function to check a square is 
// guarded by a pawn.  returns 1 if true
//
//--------------------------------------------

inline int position::pawn_guard(int tsq, int side, pawn_data *pawn_record) {
  if((1ULL<<tsq)&pawn_record->pawn_attacks[side]) return 1;
  else return 0;
}

/*--------------------------- Score position ------------------------*/
// Position is scored from point of view of white to move.  Score is
// currently based on a number of factors: Open files, king proximity,
// good/bad bishops, bishop pairs, pawn structure, king safety, etc....

int position::score_pos(game_rec *gr, ts_thread_data *tdata NNUE_ACC_DEF)
{
#if NNUE
   // -------------------------------------------------------------------
   // NNUE evaluation: bypass classical eval when the net is loaded and
   // a valid accumulator is provided.
   // -------------------------------------------------------------------
   if (nnue_available && nnue_acc) {
     tdata->eval_count++;
     // Probe score hash table (same mechanism as classical eval)
     score_rec *scores_n = score_table + (((SCORE_SIZE-1)*((hcode)&MAX_UINT))/MAX_UINT);
     if (scores_n->get_key() == hcode && (!TRAIN_EVAL)) {
       int cached = scores_n->score;
       if (scores_n->get_key() == hcode) {
         tdata->shash_count++;
         qchecks[0] = 0;
         qchecks[1] = 0;
         return wtm ? cached : -cached;
       }
     }
     // Refresh any dirty accumulator halves (king moved in this subtree)
     if (nnue_acc->dirty[WHITE] || nnue_acc->dirty[BLACK])
       nnue_init_accumulator(*nnue_acc, *this);
     int pc = 2;  // two kings
     for (int s = 0; s < 2; s++)
       for (int pt = PAWN; pt <= QUEEN; pt++)
         pc += plist[s][pt][0];
     int score = nnue_evaluate(*nnue_acc, wtm, pc);
     // qchecks: set to zero when using NNUE (no classical king-safety info)
     qchecks[0] = 0;
     qchecks[1] = 0;
     // knowledge_scale weakening still applies
     if (gr->knowledge_scale < 100) {
       int nscore = (gr->knowledge_scale * score) / 100;
       if (pcode & 1) nscore -= int(float(score) * (100.0 - float(gr->knowledge_scale)) / 100.0);
       else           nscore += int(float(score) * (100.0 - float(gr->knowledge_scale)) / 100.0);
       if (pcode & 2) nscore -= int(float(score) * (100.0 - float(gr->knowledge_scale)) / 100.0);
       else           nscore += int(float(score) * (100.0 - float(gr->knowledge_scale)) / 100.0);
       if (pcode & 4) nscore -= int(float(score) * (100.0 - float(gr->knowledge_scale)) / 100.0);
       else           nscore += int(float(score) * (100.0 - float(gr->knowledge_scale)) / 100.0);
       score = nscore;
       if (score > 9999)  score = 9999;
       if (score < -9999) score = -9999;
     }
     // Store in score hash table.
     // Hash convention: white POV.  nnue_evaluate returns stm POV, so convert.
     int16_t score_w = (int16_t)(wtm ? score : -score);
     scores_n->qchecks[0] = 0;
     scores_n->qchecks[1] = 0;
     scores_n->score = score_w;
     scores_n->set_key(hcode, score_w, 0, 0);
     // score is already from side-to-move's POV (nnue_evaluate returns stm POV)
     return score;
   }
#endif
   int score = 0, i, file, rank, sqr;
   int wattacks = 0, battacks = 0, wtropism = 0, btropism = 0;
   int wksq = plist[WHITE][KING][1];
   int bksq = plist[BLACK][KING][1];
   pawn_rec *pawns; pawn_data pawn_record;
   score_rec *scores;

   // count this evaluation
   tdata->eval_count++;

   //-----------------------------------
   // probe score hash table  
   //-----------------------------------
   scores = score_table + (((SCORE_SIZE-1)*((hcode)&MAX_UINT))/MAX_UINT);
   //   scores = score_table+(hcode&(SCORE_SIZE-1));
   if(scores->get_key() == hcode && (!TRAIN_EVAL)) {
     qchecks[0] = scores->qchecks[0];
     qchecks[1] = scores->qchecks[1];
     score = scores->score;
     // check code again to be sure nothing
     // has changed in another thread
     if(scores->get_key() == hcode) {
       tdata->shash_count++;
       if(wtm) return(score);
       else return(-score);
     } else {
       score = 0;
     }
   }

/*+++++++++++++++++++++++++++++++
|
| Score the material on the board
|
+++++++++++++++++++++++++++++++++*/
   if(!TRAIN_EVAL) {
     score = (wtm ? (material) : -(material));
   } else {
     score = 0;
     for(int piece = PAWN; piece < KING; piece++) {
       score += plist[WHITE][piece][0]*value[piece];
       score -= plist[BLACK][piece][0]*value[piece];
     }
   }

   // reset flag about whether to do checks in qsearch 
   //  -- might be set again in the score_king function below
   qchecks[0] = 0; 
   qchecks[1] = 0; 
   
/*+++++++++++++++++++++++++++++++++++++
|
| Trade Pieces, Keep Pawns when ahead
|  -- Note score is only material at this point
| 
|+++++++++++++++++++++++++++++++++++++*/
   if(score > 0) {
     int TRADES = AVERAGE_SCORE(TRADES_EARLY, TRADES_LATE);
     score += (TRADES*score*(plist[WHITE][PAWN][0]-pieces[BLACK]))/(pieces[BLACK]*100);
   } else if(score < 0) {
     int TRADES = AVERAGE_SCORE(TRADES_EARLY, TRADES_LATE);
     score += (TRADES*score*(plist[BLACK][PAWN][0]-pieces[WHITE]))/(pieces[WHITE]*100);
   }

/*+++++++++++++++++++++++++++++++
|
| Probing Pawn Hash Table
|  -- note must work with a copy
|     of the pawn record as the 
|     hashed version may be 
|     changed by another thread
+++++++++++++++++++++++++++++++++*/
   pawns = pawn_table + (((PAWN_SIZE-1)*((pcode)&MAX_UINT))/MAX_UINT);
   //pawns = pawn_table+(pcode&(PAWN_SIZE-1));
   pawn_record = pawns->data;
   if(pawns->get_key() == pcode && (!TRAIN_EVAL)) {
    score += pawn_record.score;
    tdata->phash_count++;
   } else {
     score += score_pawns(&pawn_record);
     // store results in pawn hash table
     pawns->data = pawn_record;
     pawns->set_key(pcode,pawn_record.score,pawn_record.pawn_attacks[BLACK],pawn_record.pawn_attacks[WHITE]);
   } 

/*+++++++++++++++++++++++++++++++
|
| Bonus for side on move
|
+++++++++++++++++++++++++++++++++*/
   if(wtm) {
     score += (AVERAGE_SCORE(SIDE_ON_MOVE_EARLY,SIDE_ON_MOVE_LATE));
   } else {
     score -= (AVERAGE_SCORE(SIDE_ON_MOVE_EARLY,SIDE_ON_MOVE_LATE)); 
   }

/*+++++++++++++++++++++++++++++++
|
| Examining Passed Pawns
|
+++++++++++++++++++++++++++++++++*/
   // For White
   if(pawn_record.passed_w) {
    for(file=0;file<8;file++) {
      if(BITMASK(file)&pawn_record.passed_w) { 
	for(i=file+48;i>7;i-=8) {
	  if(PTYPE(sq[i])==PAWN) {
	    rank = RANK(i);
	    //---------------------------------------------
	    // See if we have the wrong bishop to protect
	    // the promotion square of a pawn
	    //---------------------------------------------
	    if(pieces[WHITE] == 2) {
	      if(plist[WHITE][BISHOP][0] == 1 
		 && COLOR(plist[WHITE][BISHOP][1]) != COLOR(SQR(file,7))) {
		score -= (WRONG_BISHOP*(rank-1)*gstage)/40;
	      }
	    }
	    /*+++++++++++++++++++++++++++++++++
	      |
	      | Outside passed pawns
	      |
	      +++++++++++++++++++++++++++++++++++*/ 
	    if(file > FILE(bksq)+3 || file < FILE(bksq)-3)
	      score += (OUTSIDE_PASSED_PAWN*(rank-1)*gstage)/(4*pieces[BLACK]);
	    /*+++++++++++++++++++++++++++++++++
	      |
	      | If no material: Can King Catch?
	      |
	      +++++++++++++++++++++++++++++++++++*/ 
	    //if(pieces[BLACK] == 1) {
	    if((7-rank < taxi_cab[file+56][bksq] && wtm)
	       || (7-rank < taxi_cab[file+56][bksq]-1)) {
	      score += (FREE_PASSED_PAWN*(rank-1)*gstage)/(4*pieces[BLACK]*pieces[BLACK]);
	    } 
	    //}           
	    break;      
	  }
	}
      }
    }
   } 
   // For Black
   if(pawn_record.passed_b) {
    for(file=0;file<8;file++) {
      if(BITMASK(file)&pawn_record.passed_b) {
	for(i=file+8;i<56;i+=8) {
	  if(PTYPE(sq[i])==PAWN) {
	    rank = RANK(i);
	    //---------------------------------------------
	    // See if we have the wrong bishop to protect
	    // the promotion square of a pawn
	    //---------------------------------------------
	    if(pieces[BLACK] == 2) {
	      if(plist[BLACK][BISHOP][0] == 1 
		 && COLOR(plist[BLACK][BISHOP][1]) != COLOR(SQR(file,0))) {
		score += (WRONG_BISHOP*(6-rank)*gstage)/40;
	      }
	    }
	    /*+++++++++++++++++++++++++++++++++
	      |
	      | Outside passed pawns
	      |
	      +++++++++++++++++++++++++++++++++++*/ 
	    if(file > FILE(wksq)+3 || file < FILE(wksq)-3)
	      score -= (OUTSIDE_PASSED_PAWN*(6-rank)*gstage)/(4*pieces[WHITE]);
	    /*+++++++++++++++++++++++++++++++++
	      |
	      | If no material: Can King Catch?
	      |
	      +++++++++++++++++++++++++++++++++++*/ 
	    //if(pieces[WHITE] == 1) {
	    if((rank < taxi_cab[file][wksq] && (wtm^1))
	       || (rank < taxi_cab[file][wksq]-1)) {
	      score -= (FREE_PASSED_PAWN*(6-rank)*gstage)/(4*pieces[WHITE]*pieces[WHITE]);
	    }
	    //}   
	    break;      
	  }
	}
      }
    }
   }    

   /**********************************
   |
   | Add up pawn attacks near king
   |
   ++++++++++++++++++++++++++++++++++*/
   // -- attacks on squares adjacent to black king
   if(pawn_record.pawn_attacks[WHITE] && plist[WHITE][QUEEN][0]) {
     if(RANK(bksq) > 2) battacks += pawn_guard(bksq-8, WHITE, &pawn_record);
     if(RANK(bksq) < 6) battacks += pawn_guard(bksq+8, WHITE, &pawn_record);
     if(FILE(bksq)) {
       battacks += pawn_guard(bksq-1,WHITE,&pawn_record);
       if(RANK(bksq) > 2) battacks += pawn_guard(bksq-9, WHITE, &pawn_record);
       if(RANK(bksq) < 6) battacks += pawn_guard(bksq+7, WHITE, &pawn_record);
     }
     if(FILE(bksq) < 7) {
       battacks += pawn_guard(bksq+1,WHITE,&pawn_record);
       if(RANK(bksq) > 2) battacks += pawn_guard(bksq-7, WHITE, &pawn_record);
       if(RANK(bksq) < 6) battacks += pawn_guard(bksq+9, WHITE, &pawn_record);
     }
   }
   // -- attacks on squares adjacent to white king
   if(pawn_record.pawn_attacks[BLACK] && plist[BLACK][QUEEN][0]) {
     if(RANK(wksq) > 1) wattacks += pawn_guard(wksq-8, BLACK, &pawn_record);
     if(RANK(wksq) < 5) wattacks += pawn_guard(wksq+8, BLACK, &pawn_record);
     if(FILE(wksq)) {
       wattacks += pawn_guard(wksq-1,BLACK,&pawn_record);
       if(RANK(wksq) > 1) wattacks += pawn_guard(wksq-9, BLACK, &pawn_record);
       if(RANK(wksq) < 5) wattacks += pawn_guard(wksq+7, BLACK, &pawn_record);
     }
     if(FILE(wksq) < 7) {
       wattacks += pawn_guard(wksq+1,WHITE,&pawn_record);
       if(RANK(wksq) > 1) wattacks += pawn_guard(wksq-7, BLACK, &pawn_record);
       if(RANK(wksq) < 5) wattacks += pawn_guard(wksq+9, BLACK, &pawn_record);
     }
   }

/*+++++++++++++++++++++++++++++++
|
| Evaluate Bishop Trap
|
+++++++++++++++++++++++++++++++++*/
    if(ID(sq[48]) == WBISHOP 
       && ID(sq[41]) == BPAWN && ID(sq[50]) == BPAWN)
       score -= value[BISHOP] - 200;
    if(ID(sq[55]) == WBISHOP 
       && ID(sq[46]) == BPAWN && ID(sq[53]) == BPAWN)
       score -= value[BISHOP] - 200;
    if(ID(sq[8]) == BBISHOP 
       && ID(sq[17]) == WPAWN && ID(sq[10]) == WPAWN)
       score += value[BISHOP] - 200;
    if(ID(sq[15]) == BBISHOP 
       && ID(sq[22]) == WPAWN && ID(sq[13]) == WPAWN)
       score += value[BISHOP] - 200;

/*+++++++++++++++++++++++++++++++++++++++++++++
|
| Basic king scoring with a tropism analysis
|
+++++++++++++++++++++++++++++++++++++++++++++++*/
   score += score_king(gr, &wtropism, &btropism);

  //----------------------------------------------------------------------------
  // define an integer for referencing piece-square tables given the game stage
  //----------------------------------------------------------------------------
  int psq_stage = int(gstage/4);
  int remainder = int(gstage%4);
  assert(psq_stage >= 0 && psq_stage <= 3);
  assert(remainder >= 0 && remainder <= 3);

/*+++++++++++++++++++++++++++++++
|
| Evaluate Knights
|
+++++++++++++++++++++++++++++++++*/
   for(i=1;i<=plist[WHITE][KNIGHT][0];i++) {
    sqr = plist[WHITE][KNIGHT][i];
    if(pawn_guard(sqr,BLACK,&pawn_record)) score -= PAWN_THREAT_MINOR;
    if(gstage < 12) { 	
      score += ((4-remainder)*piece_sq[psq_stage][KNIGHT][whitef[sqr]]
		+remainder*piece_sq[psq_stage+1][KNIGHT][whitef[sqr]])/4;
    } else {
      score += piece_sq[3][KNIGHT][whitef[sqr]];
    }
    score += (KNIGHT_MOBILITY*knight_mobility(sqr, bksq, &battacks, &pawn_record))/16;
    // score outposts
    if(RANK(sqr) > 3 && RANK(sqr) < 7 && 
       FILE(sqr) < 6 && FILE(sqr) > 1 && !pawn_guard(sqr,BLACK,&pawn_record)) {
      if(RANK(sqr) != 4 || (!(sq[sqr+9] == EMPTY && sq[sqr+17] == BPAWN) 
			    && !(sq[sqr+7] == EMPTY && sq[sqr+15] == BPAWN))) {     // pawn cannot attack sqr in near future
	if(plist[BLACK][PAWN][0] >= 4) { // minimum opponent pawn count, suggested by Ed Schroeder on CCC
	  // score bonus based on guarded status
	  if(!pawn_guard(sqr,WHITE,&pawn_record)) score += MINOR_OUTP_UNGUARD;
	  else score += MINOR_OUTPOST;
	}
	// increase bonus if blocks a pawn, suggested by Ed Schroeder
        //  -- also if one of OUR pawns is serving as a blocker, suggested by stockfish eval  
	if(PTYPE(sq[sqr+8]) == PAWN) { 
	  score += MINOR_BLOCKER;
	}
      }
    }
   }
   for(i=1;i<=plist[BLACK][KNIGHT][0];i++) { 
    sqr = plist[BLACK][KNIGHT][i];
    if(pawn_guard(sqr,WHITE,&pawn_record)) score += PAWN_THREAT_MINOR;
    if(gstage < 12) { 	
      score -= ((4-remainder)*piece_sq[psq_stage][KNIGHT][sqr]
		+remainder*piece_sq[psq_stage+1][KNIGHT][sqr])/4;
    } else {
      score -= piece_sq[3][KNIGHT][sqr];
    }
    score -= (KNIGHT_MOBILITY*knight_mobility(sqr, wksq, &wattacks, &pawn_record))/16;
    // score outposts
    if(RANK(sqr) < 4 && RANK(sqr) > 0 &&
       FILE(sqr) < 6 && FILE(sqr) > 1 && !pawn_guard(sqr,WHITE,&pawn_record)) {
      if(RANK(sqr) != 3 || (!(sq[sqr-9] == EMPTY && sq[sqr-17] == WPAWN) 
			    && !(sq[sqr-7] == EMPTY && sq[sqr-15] == WPAWN))) {     // pawn cannot attack sqr in near future
	if(plist[WHITE][PAWN][0] >= 4) { // minimum opponent pawn count, suggested by Ed Schroeder on CCC
	  // score bonus based on guarded status
	  if(!pawn_guard(sqr,BLACK,&pawn_record)) score -= MINOR_OUTP_UNGUARD;
	  else score -= MINOR_OUTPOST;
	}
	// increase bonus if blocks a pawn, suggested by Ed Schroeder
        //  -- also if one of OUR pawns is serving as a blocker, suggested by stockfish eval  
	if(PTYPE(sq[sqr-8]) == PAWN) { 
	  score -= MINOR_BLOCKER;
	}     
      }
    }
   }

   //cout << "rscore = " << rscore << "\n";
   //cout << "score update after Knights = " << score << "\n";

/*+++++++++++++++++++++++++++++++
|
| Evaluate Bishops
|
+++++++++++++++++++++++++++++++++*/
   for(i=1;i<=plist[WHITE][BISHOP][0];i++) {
    sqr = plist[WHITE][BISHOP][i];
    if(pawn_guard(sqr,BLACK,&pawn_record)) score -= PAWN_THREAT_MINOR;
    if(gstage < 12) { 	
      score += ((4-remainder)*piece_sq[psq_stage][BISHOP][whitef[sqr]]
		+remainder*piece_sq[psq_stage+1][BISHOP][whitef[sqr]])/4;
    } else {
      score += piece_sq[3][BISHOP][whitef[sqr]];
    }
    if(i==2) score += BISHOP_PAIR;
    score += (BISHOP_MOBILITY*bishop_mobility(sqr, bksq, &battacks, &pawn_record))/16;    
    // score outposts
    if(RANK(sqr) > 3 && RANK(sqr) < 7 && 
       FILE(sqr) < 6 && FILE(sqr) > 1 && !pawn_guard(sqr,BLACK,&pawn_record)) {
      if(RANK(sqr) != 4 || (!(sq[sqr+9] == EMPTY && sq[sqr+17] == BPAWN) 
			    && !(sq[sqr+7] == EMPTY && sq[sqr+15] == BPAWN))) {     // pawn cannot attack sqr in near future
	if(plist[BLACK][PAWN][0] >= 4) { // minimum opponent pawn count, suggested by Ed Schroeder on CCC
	  // score bonus based on guarded status
	  if(!pawn_guard(sqr,WHITE,&pawn_record)) score += BMINOR_OUTP_UNGUARD;
	  else score += BMINOR_OUTPOST;
	}
	// increase bonus if blocks a pawn, suggested by Ed Schroeder
        //  -- also if one of OUR pawns is serving as a blocker, suggested by stockfish eval  
	if(PTYPE(sq[sqr+8]) == PAWN) { 
	  score += BMINOR_BLOCKER;
	}
      }
    }
   }
   for(i=1;i<=plist[BLACK][BISHOP][0];i++) {
    sqr = plist[BLACK][BISHOP][i];
    if(pawn_guard(sqr,WHITE,&pawn_record)) score += PAWN_THREAT_MINOR;
    if(gstage < 12) { 	
      score -= ((4-remainder)*piece_sq[psq_stage][BISHOP][sqr]
		+remainder*piece_sq[psq_stage+1][BISHOP][sqr])/4;
    } else {
      score -= piece_sq[3][BISHOP][sqr];
    }
    if(i==2) score -= BISHOP_PAIR;
    score -= (BISHOP_MOBILITY*bishop_mobility(sqr, wksq, &wattacks, &pawn_record))/16;
    // score outposts
    if(RANK(sqr) < 4 && RANK(sqr) > 0 && 
       FILE(sqr) < 6 && FILE(sqr) > 1 && !pawn_guard(sqr,WHITE,&pawn_record)) {
      if(RANK(sqr) != 3 || (!(sq[sqr-9] == EMPTY && sq[sqr-17] == WPAWN) 
			    && !(sq[sqr-7] == EMPTY && sq[sqr-15] == WPAWN))) {     // pawn cannot attack sqr in near future
	if(plist[WHITE][PAWN][0] >= 4) { // minimum opponent pawn count, suggested by Ed Schroeder on CCC
	  // score bonus based on guarded status
	  if(!pawn_guard(sqr,BLACK,&pawn_record)) score -= BMINOR_OUTP_UNGUARD;
	  else score -= BMINOR_OUTPOST;
	}
	// increase bonus if blocks a pawn, suggested by Ed Schroeder
        //  -- also if one of OUR pawns is serving as a blocker, suggested by stockfish eval 
	if(PTYPE(sq[sqr-8]) == PAWN) { 
	  score -= BMINOR_BLOCKER;
	}
      }    
    }
   }

   //cout << "score update after Bishops = " << score << "\n";

/*+++++++++++++++++++++++++++++++
|
| Evaluate Rooks
|
+++++++++++++++++++++++++++++++++*/
   
   for(i=1;i<=plist[WHITE][ROOK][0];i++) {
    sqr = plist[WHITE][ROOK][i]; 
    if(pawn_guard(sqr,BLACK,&pawn_record)) score -= PAWN_THREAT_MAJOR;
    if(gstage < 12) { 	
      score += ((4-remainder)*piece_sq[psq_stage][ROOK][whitef[sqr]]
		+remainder*piece_sq[psq_stage+1][ROOK][whitef[sqr]])/4;
    } else {
      score += piece_sq[3][ROOK][whitef[sqr]];
    }
//    if(i == 2 && rank != 0 && (rank == RANK(sqr) || file == FILE(sqr))) {
//      score += CONNECTED_ROOKS*(12-psq_stage)/12;
//    }
    rank = RANK(sqr); file = FILE(sqr);
    score += (ROOK_MOBILITY*rook_mobility(sqr, bksq, &battacks, &pawn_record))/16;
    //-----------------------------
    // Rook on 7TH RANK
    //-----------------------------
    //if(rank == 6 && bksq > 48) {
    //  score += ROOK_ON_7TH;   
    //}    
    /*+++++++++++++++++++++++++++++++
    |
    | Rooks on open and half-open files
    |
    +++++++++++++++++++++++++++++++++*/
    if(pawn_record.open_files&BITMASK(file)) {
      score += ROOK_OPEN_FILE; 
    } else if(pawn_record.half_open_files_w&BITMASK(file)) {
      score += ROOK_HALF_OPEN_FILE;
    }
    /*+++++++++++++++++++++++++++++++
    |
    | Rooks boxed in by the king
    |
    +++++++++++++++++++++++++++++++++*/
    if(gstage < 10 
        && rank < 3 && RANK(wksq) == 0 
        &&((file > 4 && FILE(wksq) > 5 && FILE(wksq) < file
            && (sq[14] == WPAWN || sq[22] == WPAWN)
            && (sq[15] == WPAWN || sq[23] == WPAWN))  
           || 
           (file < 3 && FILE(wksq) < 2 && FILE(wksq) > file
            && (sq[8] == WPAWN || sq[16] == WPAWN)
            && (sq[9] == WPAWN || sq[17] == WPAWN))  
           )) 
     score -= BOXED_IN_ROOK;
   }

   for(i=1;i<=plist[BLACK][ROOK][0];i++) {
    sqr = plist[BLACK][ROOK][i]; 
    if(pawn_guard(sqr,WHITE,&pawn_record)) score += PAWN_THREAT_MAJOR;
    if(gstage < 12) { 	
      score -= ((4-remainder)*piece_sq[psq_stage][ROOK][sqr]
		+remainder*piece_sq[psq_stage+1][ROOK][sqr])/4;
    } else {
      score -= piece_sq[3][ROOK][sqr];
    }
//    if(i == 2 && rank != 0 && (rank == RANK(sqr) || file == FILE(sqr))) {
//      score -= CONNECTED_ROOKS*(12-psq_stage)/12;
//    }
    rank = RANK(sqr); file = FILE(sqr);
    score -= (ROOK_MOBILITY*rook_mobility(sqr, wksq, &wattacks, &pawn_record))/16;
    //-----------------------------
    // Rook on 7TH RANK
    //-----------------------------
    //if(rank == 1 && wksq < 16) {
    //  score -= ROOK_ON_7TH;   
    //}    
    /*+++++++++++++++++++++++++++++++
    |
    | Rooks on open and half-open files
    |
    +++++++++++++++++++++++++++++++++*/
    if(pawn_record.open_files&BITMASK(file)) {
      score -= ROOK_OPEN_FILE;
    } else if(pawn_record.half_open_files_b&BITMASK(file)) {
      score -= ROOK_HALF_OPEN_FILE;
    }
    /*+++++++++++++++++++++++++++++++
    |
    | Rooks boxed in by the king
    |
    +++++++++++++++++++++++++++++++++*/
    if(gstage < 10 
        && rank > 5 && RANK(bksq) == 7 
        &&((file > 4 && FILE(bksq) > 5 && FILE(bksq) < file
            && (sq[54] == BPAWN || sq[46] == BPAWN)
            && (sq[55] == BPAWN || sq[47] == BPAWN))  
           || 
           (file < 3 && FILE(bksq) < 2 && FILE(bksq) > file
            && (sq[48] == BPAWN || sq[40] == BPAWN)
            && (sq[49] == BPAWN || sq[41] == BPAWN))  
           )) 
     score += BOXED_IN_ROOK;
   }

   //cout << "score update after Rooks = " << score << "\n";

/*+++++++++++++++++++++++++++++++
|
| Evaluate Queens
|
+++++++++++++++++++++++++++++++++*/
   for(i=1;i<=plist[WHITE][QUEEN][0];i++) {
    sqr = plist[WHITE][QUEEN][i]; 
    if(pawn_guard(sqr,BLACK,&pawn_record)) score -= PAWN_THREAT_MAJOR;
    if(gstage < 12) { 	
      score += ((4-remainder)*piece_sq[psq_stage][QUEEN][whitef[sqr]]
		+remainder*piece_sq[psq_stage+1][QUEEN][whitef[sqr]])/4;
    } else {
      score += piece_sq[3][QUEEN][whitef[sqr]];
    }
    score += (QUEEN_MOBILITY*(bishop_mobility(sqr,bksq,&battacks, &pawn_record)
			      +rook_mobility(sqr,bksq,&battacks, &pawn_record)))/16;
   }
   for(i=1;i<=plist[BLACK][QUEEN][0];i++) { 
    sqr = plist[BLACK][QUEEN][i];
    if(pawn_guard(sqr,WHITE,&pawn_record)) score += PAWN_THREAT_MAJOR;
    if(gstage < 12) { 	
      score -= ((4-remainder)*piece_sq[psq_stage][QUEEN][sqr]
		+remainder*piece_sq[psq_stage+1][QUEEN][sqr])/4;
    } else {
      score -= piece_sq[3][QUEEN][sqr];
    }
    score -= (QUEEN_MOBILITY*(bishop_mobility(sqr,wksq,&wattacks, &pawn_record)
			      +rook_mobility(sqr,wksq,&wattacks, &pawn_record)))/16;
   }

   //cout << "score update after Queens = " << score << "\n";

/*+++++++++++++++++++++++++++++++
|
| Score for Attacks on King
|
+++++++++++++++++++++++++++++++++*/
  
   if(gstage < KING_SAFETY_STAGE+4) {

     if(wtropism+wattacks > 0) wattacks = MIN((wattacks+wtropism), 15);
     else wattacks = 0;
     if(btropism+battacks > 0) battacks = MIN((battacks+btropism), 15);
     else battacks = 0; 

     // set flag about whether to do checks in qsearch 
     if(gstage < KING_SAFETY_STAGE) { 
       qchecks[0] = wtropism; // (wattacks+wtropism)/2;
       qchecks[1] = btropism; // (battacks+btropism)/2;
     }

     // evaluate pawn shelter for king
     int wdefects=1;
     int bdefects=1;

     if(gstage < KING_SAFETY_STAGE) {
       if(plist[BLACK][QUEEN][0]) {
	 if(wksq > 39) wdefects=6; 
	 else {
	   if(!FILE(wksq)) wdefects += 2;
	   else {
	     if(sq[wksq+7] != WPAWN) { 
	       wdefects += 1;
	       if(sq[wksq+15] != WPAWN) wdefects += 1;
	     }
	   }
	   if(FILE(wksq) == 7) wdefects += 2;
	   else {
	     if(sq[wksq+9] != WPAWN) { 
	       wdefects += 1;
	       if(sq[wksq+17] != WPAWN) wdefects += 1;
	     }
	   }
	   if(sq[wksq+8] != WPAWN) { 
	     wdefects += 1;
	     if(sq[wksq+16] != WPAWN) wdefects += 1;
	   }
	 }
       }

       if(plist[WHITE][QUEEN][0]) {
	 if(bksq < 24) bdefects=6; 
	 else {
	   if(!FILE(bksq)) bdefects += 2;
	   else {
	     if(sq[bksq-9] != BPAWN) { 
	       bdefects += 1;
	       if(sq[bksq-17] != BPAWN) bdefects += 1;
	     }
	   }
	   if(FILE(bksq) == 7) bdefects += 2;
	   else {
	     if(sq[bksq-7] != BPAWN) { 
	       bdefects += 1;
	       if(sq[bksq-15] != BPAWN) bdefects += 1;
	     }
	   }
	   if(sq[bksq-8] != BPAWN) { 
	     bdefects += 1;
	     if(sq[bksq-16] != BPAWN) bdefects += 1;
	   }
	 }
       }

       // Add defects for open/half-open files near king
       if(plist[BLACK][QUEEN][0] && plist[BLACK][ROOK][0]) {
	 int wkfile = FILE(wksq);
	 if(pawn_record.half_open_files_b&BITMASK(wkfile)) wdefects += 1; 
	 if(pawn_record.open_files&BITMASK(wkfile)) wdefects += 1;
	 if(wkfile < 7) {
	   if(pawn_record.half_open_files_b&BITMASK(wkfile+1)) wdefects += 1; 
	   if(pawn_record.open_files&BITMASK(wkfile+1)) wdefects += 1;
	 }
	 if(wkfile > 0) {
	   if(pawn_record.half_open_files_b&BITMASK(wkfile-1)) wdefects += 1; 
	   if(pawn_record.open_files&BITMASK(wkfile-1)) wdefects += 1;
	 }
       }
       if(plist[WHITE][QUEEN][0] && plist[WHITE][ROOK][0]) {
	 int bkfile = FILE(bksq);
	 if(pawn_record.half_open_files_w&BITMASK(bkfile)) bdefects += 1; 
	 if(pawn_record.open_files&BITMASK(bkfile)) bdefects += 1;
	 if(bkfile < 7) {
	   if(pawn_record.half_open_files_w&BITMASK(bkfile+1)) bdefects += 1; 
	   if(pawn_record.open_files&BITMASK(bkfile+1)) bdefects += 1;
	 }
 	 if(bkfile > 0) {
	   if(pawn_record.half_open_files_w&BITMASK(bkfile-1)) bdefects += 1; 
	   if(pawn_record.open_files&BITMASK(bkfile-1)) bdefects += 1;
	 }
       }
       
       if(gstage > KING_SAFETY_STAGE-4) {
	 wdefects = MAX(1,wdefects-4+KING_SAFETY_STAGE-gstage);
	 bdefects = MAX(1,bdefects-4+KING_SAFETY_STAGE-gstage);
       }

     }

     if(!plist[WHITE][QUEEN][0]) { battacks = 0; btropism = 0; bdefects = 0; }
     if(!plist[BLACK][QUEEN][0]) { wattacks = 0; wtropism = 0; wdefects = 0; }

     // attack score from white point of view
     int attack_score = (-attack_scale[wattacks]*sqrt(1.0+4.95*wdefects)
			 +attack_scale[battacks]*sqrt(1.0+4.95*bdefects))*2.74;  //2.46;

     // adjust for structural problems (not multiplied by attacks)
     if(gstage < KING_SAFETY_STAGE) { attack_score += ((-wdefects+bdefects)*3.1*KING_SAFETY_STAGE)/(gstage+1); }

     // scaling if we are near the stage of the game where we no longer account for this   
     if(gstage <= KING_SAFETY_STAGE) { score += attack_score; } 
     else if(gstage < KING_SAFETY_STAGE+4) {
       score += ((attack_score*(KING_SAFETY_STAGE+4 - gstage))/4);
     }

   }

   /*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   | Detect likely && actual draws
   +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
   if(score > 0 && !plist[WHITE][PAWN][0] && !plist[WHITE][QUEEN][0]) {
     // if we have less than 5 pieces and our material piece advantage is less
     //  a rook, then it is likely a draw  -- 50 cp margin accounts for BISHOP/KNIGHT diff
     if(pieces[WHITE] < 5 && ABS(material) < value[ROOK]-50-plist[BLACK][PAWN][0]*value[PAWN])
       score = score/8;
     // if we have less than 4 pieces and no rook and not 2 bishops, it *is* a draw
     if(pieces[WHITE] < 4 && (!plist[WHITE][BISHOP][0] || pieces[WHITE] < 3) 
      && !plist[WHITE][ROOK][0]) score = 0;
   }
   if(score < 0 && !plist[BLACK][PAWN][0] && !plist[BLACK][QUEEN][0]) {
     // if we have less than 5 pieces and our material piece advantage is less
     //  a rook, then it is likely a draw -- 50 cp margin accounts for BISHOP/KNIGHT diff
     if(pieces[BLACK] < 5 && ABS(material) < value[ROOK]-50-plist[WHITE][PAWN][0]*value[PAWN])
       score = score/8;
     // if we have less than 4 pieces and no rook and not 2 bishops, it *is* a draw
     if(pieces[BLACK] < 4 && (!plist[BLACK][BISHOP][0] || pieces[BLACK] < 3) 
      && !plist[BLACK][ROOK][0]) score = 0;
   }

   /* make a knowledge adjustment to weaken play */
   if(gr->knowledge_scale < 100) {
     int nscore = (gr->knowledge_scale*score)/100;
     if(pcode&1) nscore -= int(float(score)*(100.0-float(gr->knowledge_scale))/100.0);
     else nscore += int(float(score)*(100.0-float(gr->knowledge_scale))/100.0);
     if(pcode&2) nscore -= int(float(score)*(100.0-float(gr->knowledge_scale))/100.0);
     else nscore += int(float(score)*(100.0-float(gr->knowledge_scale))/100.0);
     if(pcode&4) nscore -= int(float(score)*(100.0-float(gr->knowledge_scale))/100.0);
     else nscore += int(float(score)*(100.0-float(gr->knowledge_scale))/100.0);
     score = nscore;
     if(score > 9999) score = 9999;
     if(score < -9999) score = -9999;
   }

#if USE_MTD
   //-------------------------------------------------
   // setup a coarse granular eval for use in MTD(f)
   //-------------------------------------------------
   score /= EVAL_GRAN;
   score *= EVAL_GRAN;
#endif

   //-----------------------------------
   // store values in score hash table
   //-----------------------------------
   scores->qchecks[0] = char(qchecks[0]);
   scores->qchecks[1] = char(qchecks[1]);
   scores->score = int16_t(score);
   scores->set_key(hcode, score, qchecks[0], qchecks[1]);

   // check values for violations of assumptions
   assert(ABS(score) < 30000);
   assert(ABS(qchecks[0]) < 255);
   assert(ABS(qchecks[1]) < 255);

   if(wtm) return(score);
   else return(-score);

}


/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
|
|
|  Score King:  Function to Examine the King and King Safety 
|
|
|
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

int position::score_king(game_rec *gr, int *wtropism, int *btropism)
{
   int score = 0;
   int wksq = plist[WHITE][KING][1];
   int bksq = plist[BLACK][KING][1];

   /*+++++++++++++++++++++++++++++++
   |
   | Add values from piece-square table
   |
   +++++++++++++++++++++++++++++++++*/   
   if(gstage < 12) { 	
     //----------------------------------------------------------------------------
     // define an integer for referencing piece-square tables given the game stage
     //----------------------------------------------------------------------------
     int psq_stage = int(gstage/4);
     int remainder = int(gstage%4);
     assert(psq_stage >= 0 && psq_stage <= 3);
     assert(remainder >= 0 && remainder <= 3);

     score += ((4-remainder)*piece_sq[psq_stage][KING][whitef[wksq]]
	       +remainder*piece_sq[psq_stage+1][KING][whitef[wksq]])/4;
     score -= ((4-remainder)*piece_sq[psq_stage][KING][bksq]
	       +remainder*piece_sq[psq_stage+1][KING][bksq])/4;
   } else {
     score += piece_sq[3][KING][whitef[wksq]];
     score -= piece_sq[3][KING][bksq];
   }

   /*+++++++++++++++++++++++++++++++
   |
   | Examine Castling 
   |
   +++++++++++++++++++++++++++++++++*/   

   if(gstage < KING_SAFETY_STAGE) {

     int castle_score = 0;

     /*+++++++++++++++++++++++++++++++
     |
     | Examine White Castling Structure 
     |
     +++++++++++++++++++++++++++++++++*/
     if(RANK(wksq) || 
	!((FILE(wksq) > 5 && sq[7] != WROOK && sq[6] != WROOK) || 
	  (FILE(wksq) < 4 && sq[0] != WROOK && sq[1] != WROOK && sq[2] != WROOK))) {
       castle_score -= CASTLED;      
       if(!(castle&3)) castle_score -= NO_POSSIBLE_CASTLE;
     } 

     /*+++++++++++++++++++++++++++++++
     |
     | Examine Black Castling Structure 
     |
     +++++++++++++++++++++++++++++++++*/   
	if(RANK(bksq) < 7 || 
	   !((FILE(bksq) > 5 && sq[63] != BROOK && sq[62] != BROOK) || 
	     (FILE(bksq) < 4 && sq[56] != BROOK && sq[57] != BROOK && sq[58] != BROOK))) {
       castle_score += CASTLED;
       if(!(castle&12)) castle_score += NO_POSSIBLE_CASTLE;
     } 

     // scaling if we are near the stage of the game where we no longer account for this   
     if(gstage <= KING_SAFETY_STAGE-4) score += castle_score;
     else if(gstage < 10) {
       score += ((castle_score*(KING_SAFETY_STAGE - gstage))/4);
     }

   }

   //-------------------------------------------------------------
   // simple calculation of king tropism for later use
   // in determining effect of king safety to due attacks near king
   //   NOTE: tropism scores only increased if queen is on the board
   //-------------------------------------------------------------
   int i,j;
   for(j=KNIGHT; j <= QUEEN; j++) {                   // start of piece loop
     for(i=1; i<=plist[BLACK][j][0];i++) {            // count black pieces
       if(plist[BLACK][QUEEN][0]) {                   
	 if(taxi_cab[plist[BLACK][j][i]][wksq] < 4) { 
	   (*wtropism)++;
	   if(j == QUEEN) (*wtropism) += 1;
	 }
       }
       if(taxi_cab[plist[BLACK][j][i]][bksq] < 3) (*btropism)--;
     }
     
     for(i=1; i<=plist[WHITE][j][0];i++) {            // count white pieces  
       if(plist[WHITE][QUEEN][0]) {
	 if(taxi_cab[plist[WHITE][j][i]][bksq] < 4) { 
	   (*btropism)++;
	   if(j == QUEEN) (*btropism) += 1;
	 }
       }
       if(taxi_cab[plist[WHITE][j][i]][wksq] < 3) (*wtropism)--;
     }
   }                                                 // end of piece loop 

   // don't allow negative tropism scores 
   //if((*wtropism) < 0) { (*wtropism) = 0; }  
   //if((*btropism) < 0) { (*btropism) = 0; } 
   
   return score;

}

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
|
|
|  Score Pawns:  Function to Examine Pawn structure 
|                and hash the results. 
|
|
|
|
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
#define wpawn  plist[WHITE][PAWN]
#define bpawn  plist[BLACK][PAWN]

int position::score_pawns(pawn_data *pawn_record)
{
   int score = 0, i, j;
   int wpawn_count = plist[WHITE][PAWN][0];
   int bpawn_count = plist[BLACK][PAWN][0];
   int psq, psq2, file, rank;
   int weak, isle, passed;
   unsigned int last_passed;
   int space = 0;

/*+++++++++++++++++++++++++++++++
|
| Setting the pawn hash parameters
|
+++++++++++++++++++++++++++++++++*/
   pawn_record->pawn_attacks[BLACK] = 0ULL;
   pawn_record->pawn_attacks[WHITE] = 0ULL;
   pawn_record->open_files = 255;
   pawn_record->half_open_files_w = 255;
   pawn_record->half_open_files_b = 255;
   pawn_record->passed_w = 0;
   pawn_record->passed_b = 0;

  int DOUBLED_PAWN = AVERAGE_SCORE(DOUBLED_PAWN_EARLY,DOUBLED_PAWN_LATE);
  int WEAK_PAWN = AVERAGE_SCORE(WEAK_PAWN_EARLY,WEAK_PAWN_LATE);
  int BACKWARD_PAWN = AVERAGE_SCORE(BACKWARD_PAWN_EARLY,BACKWARD_PAWN_LATE);
  int PAWN_ISLAND = AVERAGE_SCORE(PAWN_ISLAND_EARLY,PAWN_ISLAND_LATE);

  //----------------------------------------------------------------------------
  // define an integer for referencing piece-square tables given the game stage
  //----------------------------------------------------------------------------
  int psq_stage = int(gstage/4);
  int remainder = int(gstage%4);
  assert(psq_stage >= 0 && psq_stage <= 3);
  assert(remainder >= 0 && remainder <= 3);

/*+++++++++++++++++++++++++++++++
|
| First Loop Through White Pawns
|
+++++++++++++++++++++++++++++++++*/
    last_passed = 0;
    for(i = 1; i <= wpawn_count; i++) {
      /*+++++++++++++++++++++++++++++++
      |
      | Setting up variables and flags
      | -General strategy with the flags
      |  is to assume that a pawn is weak,
      |  on an island, or passed until
      |  proven otherwise.
      |
      +++++++++++++++++++++++++++++++++*/
      psq = wpawn[i];
      file = FILE(psq); rank = RANK(psq);
      isle = 1; passed = 1; weak = 1;
      /*+++++++++++++++++++++++++++++++++
      |
      |  Setup the attacked squares
      |
      +++++++++++++++++++++++++++++++++*/
      if(file < 7)
	pawn_record->pawn_attacks[WHITE] |= (1ULL<<(psq+9));
      if(file)
	pawn_record->pawn_attacks[WHITE] |= (1ULL<<(psq+7));      
      /*+++++++++++++++++++++++++++++++
      |
      | Add value from piece-square table
      |  -- interpolate values
      |
      +++++++++++++++++++++++++++++++++*/
      if(gstage < 12) { 	
	score += ((4-remainder)*piece_sq[psq_stage][PAWN][whitef[psq]]
		   +remainder*piece_sq[psq_stage+1][PAWN][whitef[psq]])/4;
      } else {
	score += piece_sq[3][PAWN][whitef[psq]];
      }
      /*+++++++++++++++++++++++++++++++
      |
      | This file is not open.
      |  - It is also not half-open for white
      |
      +++++++++++++++++++++++++++++++++*/
      if(pawn_record->open_files&BITMASK(file))
        pawn_record->open_files ^= BITMASK(file);
      if(pawn_record->half_open_files_w&BITMASK(file))
        pawn_record->half_open_files_w ^= BITMASK(file);
      /*+++++++++++++++++++++++++++++++
      |
      | Look for an immediate neighbor 
      | If one exists, add a bonus for
      | a pawn duo.
      |
      +++++++++++++++++++++++++++++++++*/
      if(rank > 1 && file < 7 && ID(sq[psq+1]) == WPAWN) score += PAWN_DUO;
      /*+++++++++++++++++++++++++++++++
      |
      | Look at my own pawns to see if we 
      | are doubled or have an island
      | also look for weak pawns 
      |
      +++++++++++++++++++++++++++++++++*/
      for(j = 1; j <= wpawn_count; j++) {
       psq2 = wpawn[j];
       if(FILE(psq2) == file) { 
         if(j > i) score -= DOUBLED_PAWN;   // only count doubled once
         if(RANK(psq2) > rank) passed = 0;  // not passed if second on file 
       }
       if(j != i) {
	 if(FILE(psq2) == file+1 || FILE(psq2) == file-1) {
	   isle = 0;
	   if(RANK(psq2) == rank || RANK(psq2) == rank-1) {
	     weak = 0;
	   }
	 }
       }
      }
      // Count "SPACE" controlled by pawn
      if(wpawn_count + bpawn_count > 11) {
	if(!weak && !isle) {
	  score += ((rank+1)*(rank+1)*SPACE)/40;
	}
      }
      /*+++++++++++++++++++++++++++++++
      |
      | If the pawn is weak, score it
      | and see if it is backward. 
      |  - Note: I need to think about
      |          doubled pawn case.
      |  - Note: Open files are automatically
      |           half open - take care with
      |           this in scoring them
      |
      +++++++++++++++++++++++++++++++++*/
      if(weak && !isle) {
       score -= WEAK_PAWN;
       if((rank < 4 && file && ID(sq[psq+15]) == BPAWN) ||
          (rank < 4 && file < 7 && ID(sq[psq+17]) == BPAWN) || 
	  (rank < 6 && ID(sq[psq+8]) == BPAWN)) {
          score -= BACKWARD_PAWN;
       }
      } else if(!weak) {   // not half open for black
	if(pawn_record->half_open_files_b&BITMASK(file))
	  pawn_record->half_open_files_b ^= BITMASK(file);
      }
      /*+++++++++++++++++++++++++++++++
      |
      | If the pawn is isolated, score it
      |
      +++++++++++++++++++++++++++++++++*/
      if(isle) {
        score -= PAWN_ISLAND;
      }
      /*+++++++++++++++++++++++++++++++
      |
      | Look at enemy pawns to see if 
      | this pawn is passed.
      |
      +++++++++++++++++++++++++++++++++*/
      for(j = 1; j <= bpawn_count; j++) {
       if(RANK(bpawn[j]) > rank && (FILE(bpawn[j]) == file ||
           FILE(bpawn[j]) == file-1 || FILE(bpawn[j]) == file+1))
         { passed = 0; break; }
      }
      /*+++++++++++++++++++++++++++++++
      |
      | If the pawn is passed, score it
      | and look for connected passers.
      | Add the file of this pawn to 
      | the passed pawn list.
      |
      +++++++++++++++++++++++++++++++++*/
      if(passed) {
        score += (PASSED_PAWN*(rank-1)*gstage)/40;
        pawn_record->passed_w |= BITMASK(file);
        // is it protected? - if so, score as if stage = 3
        if(gstage < 12) {
         if((file < 7 && ID(sq[psq-7]) == WPAWN) ||
            (file && ID(sq[psq-9]) == WPAWN))
           score += (PASSED_PAWN*(rank-1)*(12-gstage))/40; 
        } 
        // detect connected passers
        if(gstage > 6) {
          if(rank > 4) {
           if(last_passed) {
             if((file && (BITMASK(file-1)&last_passed)) ||
                (file < 7 && (BITMASK(file+1)&last_passed)))
               score += (CON_PASSED_PAWNS*(gstage - 6))/4;
           }
           last_passed |= BITMASK(file);
          }
        }
      }
    }

/*+++++++++++++++++++++++++++++++
|
| Loop Through Black Pawns
|
+++++++++++++++++++++++++++++++++*/
    last_passed = 0;
    for(i = 1; i <= bpawn_count; i++) {
      /*+++++++++++++++++++++++++++++++
      |
      | Setting up variables and flags
      | -General strategy with the flags
      |  is to assume that a pawn is weak,
      |  on an island, or passed until
      |  proven otherwise.
      |
      +++++++++++++++++++++++++++++++++*/
      psq = bpawn[i];
      file = FILE(psq); rank = RANK(psq);
      isle = 1; passed = 1; weak = 1;     
      /*+++++++++++++++++++++++++++++++++
      |
      |  Setup the attacked squares
      |
      +++++++++++++++++++++++++++++++++*/
      if(file < 7)
	pawn_record->pawn_attacks[BLACK] |= (1ULL<<(psq-7));
      if(file)
	pawn_record->pawn_attacks[BLACK] |= (1ULL<<(psq-9));      
      /*+++++++++++++++++++++++++++++++
      |
      | Add value from piece-square table, interpolate between stages
      |
      +++++++++++++++++++++++++++++++++*/
      if(gstage < 12) { 	
	score -= ((4-remainder)*piece_sq[psq_stage][PAWN][psq]
		   +remainder*piece_sq[psq_stage+1][PAWN][psq])/4;
      } else {
	score -= piece_sq[3][PAWN][psq];
      }
      /*+++++++++++++++++++++++++++++++
      |
      | This file is not open
      |  - It is also not half open for black
      |
      +++++++++++++++++++++++++++++++++*/
      if(pawn_record->open_files&BITMASK(file))
       pawn_record->open_files ^= BITMASK(file);
      if(pawn_record->half_open_files_b&BITMASK(file))
        pawn_record->half_open_files_b ^= BITMASK(file);
      /*+++++++++++++++++++++++++++++++
      |
      | Look for an immediate neighbor 
      | If one exists, add a bonus for
      | a pawn duo.
      |
      +++++++++++++++++++++++++++++++++*/
      if(rank < 6 && file < 7 && ID(sq[psq+1]) == BPAWN) score -= PAWN_DUO;
      /*+++++++++++++++++++++++++++++++
      |
      | Look at my own pawns to see if we 
      | are doubled or have an island
      | also look for weak pawns 
      |
      +++++++++++++++++++++++++++++++++*/
      for(j = 1; j <= bpawn_count; j++) {
       psq2 = bpawn[j];
       if(FILE(psq2) == file) { 
         if(j > i) score += DOUBLED_PAWN;   // only count doubled once
         if(RANK(psq2) < rank) passed = 0;  // not passed if second on file 
       }
       if(j != i) {
	 if(FILE(psq2) == file+1 || FILE(psq2) == file-1) {
	   isle = 0;
	   if(RANK(psq2) == rank || RANK(psq2) == rank+1) {
	     weak = 0;
	   }
	 }
       }
      }
      // Count "SPACE" controlled by pawn
      if(wpawn_count + bpawn_count > 11) {
	if(!weak && !isle) {
	  score -= ((8-rank)*(8-rank)*SPACE)/40;
	}
      }
      /*+++++++++++++++++++++++++++++++
      |
      | If the pawn is weak, score it
      | and see if it is backward. 
      |  - Note: Think about doubled
      |          pawn case.
      |
      +++++++++++++++++++++++++++++++++*/
      if(weak && !isle) {
       score += WEAK_PAWN;
       if((rank > 3 && file && ID(sq[psq-17]) == WPAWN) ||
          (rank > 3 && file < 7 && ID(sq[psq-15]) == WPAWN) || 
	  (rank > 1 && ID(sq[psq-8]) == WPAWN)) {
          score += BACKWARD_PAWN;
       }
      } else if(!weak) {  // not half open for white
	if(pawn_record->half_open_files_w&BITMASK(file))
	  pawn_record->half_open_files_w ^= BITMASK(file);
      }
      /*+++++++++++++++++++++++++++++++
      |
      | If the pawn is isolated, score it
      |
      +++++++++++++++++++++++++++++++++*/
      if(isle) {
        score += PAWN_ISLAND;
      }
      /*+++++++++++++++++++++++++++++++
      |
      | Look at enemy pawns to see if 
      | this pawn is passed.
      |
      +++++++++++++++++++++++++++++++++*/
      for(j = 1; j <= wpawn_count; j++) {
       if(RANK(wpawn[j]) < rank && (FILE(wpawn[j]) == file ||
           FILE(wpawn[j]) == file-1 || FILE(wpawn[j]) == file+1))
         { passed = 0; break; }
      }
      /*+++++++++++++++++++++++++++++++
      |
      | If the pawn is passed, score it
      | and look for connected passers.
      | Add the file of this pawn to 
      | the passed pawn list.
      |
      +++++++++++++++++++++++++++++++++*/
      if(passed) {
        score -= (PASSED_PAWN*(6-rank)*gstage)/40;
        pawn_record->passed_b |= BITMASK(file);
        // is it protected? - if so, score as if stage = 3
        if(gstage < 12) {
         if((file < 7 && ID(sq[psq+9]) == BPAWN) ||
            (file && ID(sq[psq+7]) == BPAWN))
           score -= (PASSED_PAWN*(6-rank)*(12-gstage))/40; 
        } 
        // detect connected passers
        if(gstage > 6) {
          if(rank < 3) {
           if(last_passed) {
             if((file && (BITMASK(file-1)&last_passed)) ||
                (file < 7 && (BITMASK(file+1)&last_passed)))
               score -= (CON_PASSED_PAWNS*(gstage - 6))/4;
           }
           last_passed |= BITMASK(file);
          }
        }
      }
    }

    pawn_record->score = int16_t(score);

   return score;

}

#undef wpawn
#undef bpawn

//----------------------------------------------------------
//  Mobility functions are given below, note that
//   they include a bonus, in the form of increased
//   mobility count, for moves to squares containing enemy
//   pieces that are undefended by pawns.  This is a crude
//   bonus, but it seems to improve play.  It could probably
//   be tuned and improved.  Idea was inspired by a CCC
//   post by Lucas Braesch about en-prise pieces.
//----------------------------------------------------------
// define a mobility array to transform the result to
// a smaller value appropriate for scoring.  Transformation
// is roughly 16 * sqrt(index) - 40
int mobility_transform[16] = { -40, -24, -17, -12, -8, -4, -1, 2, 5, 8, 11, 13, 15, 18, 20, 22 };

/*------------------------------- Bishop Mobility --------------------------*/
int position::bishop_mobility(int sqr, int ksq, int *kattacks, pawn_data *pawn_record)
{
  int mm, nn, ii, self, tsq, count = 0;

  mm = FILE(sqr); nn = RANK(sqr);
  self = PSIDE(sq[sqr]);

  ii = 1;
  while (mm + ii <= 7 && nn + ii <= 7)
  {
   tsq = SQR((mm+ii),(nn+ii));
   if(PTYPE(sq[tsq]) != PAWN) { 
     if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
     if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > PTYPE(sq[sqr]))) {
       count++; 
       if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
     }
     if(PTYPE(sq[tsq]) && PSIDE(sq[tsq]) != self) break; 
   } else if(PSIDE(sq[tsq]) != self) { 
     if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
     if(!pawn_guard(tsq,!self,pawn_record)) { 
       count++; 
     }
     break; 
   } else break;
   ii++;
  }

  ii = 1;
  while (mm - ii >= 0 && nn + ii <= 7)
  {
   tsq = SQR((mm-ii),(nn+ii));
   if(PTYPE(sq[tsq]) != PAWN) { 
     if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
     if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > PTYPE(sq[sqr]))) {
       count++; 
       if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
     }
     if(PTYPE(sq[tsq]) && PSIDE(sq[tsq]) != self) break; 
   } else if(PSIDE(sq[tsq]) != self) { 
     if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
     if(!pawn_guard(tsq,!self,pawn_record)) {
       count++; 
     }
     break; 
   } else break;
   ii++;
  }

  ii = 1;
  while(mm + ii <= 7 && nn - ii >= 0)
  {
   tsq = SQR((mm+ii),(nn-ii));
   if(PTYPE(sq[tsq]) != PAWN) { 
     if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
     if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > PTYPE(sq[sqr]))) {
       count++; 
       if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
     }
     if(PTYPE(sq[tsq]) && PSIDE(sq[tsq]) != self) break; 
   } else if(PSIDE(sq[tsq]) != self) { 
     if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
     if(!pawn_guard(tsq,!self,pawn_record)) { 
       count++; 
     }
     break; 
   } else break;
   ii++;
  }

  ii = 1;
  while (mm - ii >= 0 && nn - ii >= 0)
  {
   tsq = SQR((mm-ii),(nn-ii));
   if(PTYPE(sq[tsq]) != PAWN) { 
     if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
     if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > PTYPE(sq[sqr]))) {
       count++; 
       if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
     }
     if(PTYPE(sq[tsq]) && PSIDE(sq[tsq]) != self) break; 
   } else if(PSIDE(sq[tsq]) != self) { 
     if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
     if(!pawn_guard(tsq,!self,pawn_record)) { 
       count++;
     }
     break; 
   } else break;
   ii++;
  }

  //if(count < 16) return int(VAR1*sqrt(count)-VAR2);
  //else return int(VAR1*sqrt(15)-VAR2+(count-15));

  if(count < 16) return mobility_transform[count];
  else return (mobility_transform[15]+(count-15));
  //return count;
}

/*------------------------------- Rook Mobility --------------------------*/
int position::rook_mobility(int sqr, int ksq, int *kattacks, pawn_data *pawn_record)
{
  int self, tsq, count = 0;

  self = PSIDE(sq[sqr]);

  if(sqr+8 <= 63) 
    for(tsq = sqr+8; tsq <= 63; tsq += 8) {
      if(PTYPE(sq[tsq]) != PAWN) { 
	if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
	if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > PTYPE(sq[sqr]))) {
	  count++; 
	  if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
	}
	if(PTYPE(sq[tsq]) && PSIDE(sq[tsq]) != self) break; 
      } else if(PSIDE(sq[tsq]) != self) { 
	if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
	if(!pawn_guard(tsq,!self,pawn_record)) {
	  count++; 
	}
	break; 
      } else break;
    }

  if(sqr-8 >= 0) 
    for(tsq = sqr-8; tsq >= 0; tsq -= 8) {
      if(PTYPE(sq[tsq]) != PAWN) { 
	if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
	if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > PTYPE(sq[sqr]))) {
	  count++; 
	  if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
	}
	if(PTYPE(sq[tsq]) && PSIDE(sq[tsq]) != self) break; 
      } else if(PSIDE(sq[tsq]) != self) { 
	if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
	if(!pawn_guard(tsq,!self,pawn_record)) {
	  count++; 
	}
	break; 
      } else break;
    }

  if(FILE(sqr) < 7) 
    for(tsq = sqr+1; FILE(tsq) <= 7 && tsq <= 63 && FILE(tsq) > 0; tsq += 1) {
      if(PTYPE(sq[tsq]) != PAWN) { 
	if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
	if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > PTYPE(sq[sqr]))) {
	  count++; 
	  if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
	}
	if(PTYPE(sq[tsq]) && PSIDE(sq[tsq]) != self) break; 
      } else if(PSIDE(sq[tsq]) != self) { 
	if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
	if(!pawn_guard(tsq,!self,pawn_record)) {
	  count++; 
	}
	break; 
      } else break;
    }

  if(FILE(sqr) > 0) 
    for(tsq = sqr-1; FILE(tsq) >= 0 && tsq >= 0 && FILE(tsq) < 7; tsq -= 1) {
      if(PTYPE(sq[tsq]) != PAWN) { 
	if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
	if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > PTYPE(sq[sqr]))) {
	  count++; 
	  if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
	} 
	if(PTYPE(sq[tsq]) && PSIDE(sq[tsq]) != self) break; 
      } else if(PSIDE(sq[tsq]) != self) { 
	if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
	if(!pawn_guard(tsq,!self,pawn_record)) { 
	  count++; 
	}
	break; 
      } else break;
    }

  //if(count > 15) cout << "count > 15!!" << sqr << " " << count << "\n";
  //if(count < 16) return int(VAR1*sqrt(count)-VAR2);
  //else return int(VAR1*sqrt(15)-VAR2+(count-15));

  if(count < 16) return mobility_transform[count];
  else return (mobility_transform[15]+(count-15));
  //return count;
}

/*------------------------------- Knight Mobility --------------------------*/
int position::knight_mobility(int sqr, int ksq, int *kattacks, pawn_data *pawn_record)
{
  int self, count = 0, tsq;  
  int rank=RANK(sqr), file=FILE(sqr);

  self = PSIDE(sq[sqr]);

  if(file < 6 && rank < 7) {
    tsq = sqr+10;
    if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
    if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > KNIGHT)) {
      if(count&1) count++; else count += 2;
      if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
    }
  }

  if(file < 6 && rank) {
    tsq = sqr-6;
    if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
    if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > KNIGHT)) {
      if(count&1) count++; else count += 2;
      if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
    }
  }

  if(file > 1 && rank < 7) {
    tsq = sqr+6;
    if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
    if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > KNIGHT)) {
      if(count&1) count++; else count += 2;
      if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
    }
  }

  if(file > 1 && rank) {
    tsq = sqr-10;
    if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
    if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > KNIGHT)) {
      if(count&1) count++; else count += 2;
      if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
    }
  }

  if(rank < 6 && file < 7) {
    tsq = sqr+17;
    if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
    if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > KNIGHT)) {
      if(count&1) count++; else count += 2;
      if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
    }
  }

  if(rank < 6 && file) {
    tsq = sqr+15;
    if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
    if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > KNIGHT)) {
      if(count&1) count++; else count += 2;
      if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
    }
  }

  if(rank > 1 && file < 7) {
    tsq = sqr-15;
    if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
    if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > KNIGHT)) {
      if(count&1) count++; else count += 2;
      if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
    }
  }

  if(rank > 1 && file) {
    tsq = sqr-17;
    if(taxi_cab[tsq][ksq] <= 1) (*kattacks) += 1+pawn_guard(tsq,self,pawn_record);
    if(!pawn_guard(tsq,!self,pawn_record) || (PSIDE(sq[tsq]) == !self && PTYPE(sq[tsq]) > KNIGHT)) {
      if(count&1) count++; else count += 2;
      if(PTYPE(sq[tsq]) > PAWN && PSIDE(sq[tsq]) != self) count += PTYPE(sq[tsq]);
    }
  }

  //if(count < 16) return int(VAR1*sqrt(count)-VAR2);
  //else return int(VAR1*sqrt(15)-VAR2+(count-15));

  if(count < 16) return mobility_transform[count];
  else return (mobility_transform[15]+(count-15));
  //return count;
}
