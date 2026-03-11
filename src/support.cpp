// EXchess source code, (c) Daniel C. Homan  1997-2013
// Released under the GNU public license, see file license.txt

/* support.cpp  - functions for position and tree search support */
//--------------------------------------------------------------------------------
//  move position::parse_move(char mstring[10], tree_search *temps)
//  void position::print_move(move pmove, char mstring[10], tree_search *temps)
//  void position::write_fen() 
//  void tree_search::log_search(int score)
//  void tree_search::search_display(int score)
//--------------------------------------------------------------------------------

#include "define.h"
#include "chess.h"
#include "funct.h"
#include "const.h"
#include "extern.h"

#include <iostream>
#include <cstdio>
#include <cstring>

//----------------------------------------------------------
// Function to parse a move from the human player
// This move is checked then checked to see if it is legal
//----------------------------------------------------------
move position::parse_move(char mstring[10], ts_thread_data *temps)
{
  int legal = 0, piece, to_sq = -1, from_sq = -1, promote = QUEEN;
  int from_file = -1, from_row = -1, match_count = 0;
  move play, mplay[4], nomove; 
  play.t = 0; nomove.t = 0;
  mplay[0].t = 0; mplay[1].t = 0; mplay[2].t = 0; mplay[3].t = 0;
  position t_pos;
  move_list list;

  // generate the legal moves
  allmoves(&list, temps);

  // Examining input move from a human player - checking for
  // errors and the legality of the move....

  // First resolve the first character - what piece is it?
  switch(mstring[0]) {
    case 'N': piece = KNIGHT; break;
    case 'B': piece = BISHOP; break;
    case 'R': piece = ROOK; break;
    case 'Q': piece = QUEEN; break;
    case 'K': piece = KING; break;
    case 'P': piece = PAWN; break;
    case 'O': piece = 10; break;        // if it is a castle
    case '0': piece = 10; break;
    case 'o': piece = 10; break;
    default : piece = PAWN;
  }

  if (piece != 10) {  // if is not a castle

    // Now cycle throught the rest of the characters looking
    // for move squares, capture signals, or promotion signals
    for(int i = 0; i < 10; i++) {
      if(!i && piece != PAWN) i++;
      switch(mstring[i]) {
        case '\0': i = 10; break;
        case 'x':
          if(CHAR_FILE(mstring[i+1]) <= 7 && CHAR_FILE(mstring[i+1]) >= 0
             && CHAR_ROW(mstring[i+2]) <= 7 && CHAR_ROW(mstring[i+2]) >= 0) {
           from_sq = to_sq;
           to_sq = SQR(CHAR_FILE(mstring[i+1]), CHAR_ROW(mstring[i+2]));
           i += 2; break;
          } else { i += 1; break; }
        case '-':
          if(CHAR_FILE(mstring[i+1]) <= 7 && CHAR_FILE(mstring[i+1]) >= 0
             && CHAR_ROW(mstring[i+2]) <= 7 && CHAR_ROW(mstring[i+2]) >= 0) {
           from_sq = to_sq;
           to_sq = SQR(CHAR_FILE(mstring[i+1]), CHAR_ROW(mstring[i+2]));
           i += 2; break;
          } else { i+= 1; break; }
        case '=':
          switch(mstring[i+1]) {
            case 'N': promote = KNIGHT; break;
            case 'B': promote = BISHOP; break;
            case 'R': promote = ROOK; break;
            case 'Q': promote = QUEEN; break;
          }
          i = 10;
          break;
        case '+': i = 10; break;
        default:
          if (to_sq == -1 && CHAR_ROW(mstring[i+1]) <= 7
               && CHAR_ROW(mstring[i+1]) >=0 && CHAR_FILE(mstring[i]) >= 0
               && CHAR_FILE(mstring[i]) <= 7) {
            to_sq = SQR(CHAR_FILE(mstring[i]), CHAR_ROW(mstring[i+1]));
            i += 1;
          } else {
            if (CHAR_ROW(mstring[i+1]) <= 7 && CHAR_ROW(mstring[i+1]) >= 0
                && CHAR_FILE(mstring[i]) <= 7 && CHAR_FILE(mstring[i]) >= 0) {
              from_sq = to_sq;
              to_sq = SQR(CHAR_FILE(mstring[i]), CHAR_ROW(mstring[i+1]));
              switch (mstring[i+2]) {
                case 'q': promote = QUEEN; i++; break;
                case 'n': promote = KNIGHT; i++; break;
                case 'b': promote = BISHOP; i++; break;
                case 'r': promote = ROOK; i++; break;
              }
              i += 1;
            } else {
             if(CHAR_FILE(mstring[i]) <= 7 && CHAR_FILE(mstring[i]) >= 0)
              from_file = CHAR_FILE(mstring[i]);
             else
              from_row = CHAR_ROW(mstring[i]);
            }
          }
          break;
      }
    }

    play.b.from = from_sq;  play.b.to = to_sq;

    // identify a castle from algebraic strings
    if(from_sq > -1 && PTYPE(sq[from_sq]) == KING && 
       ((FILE(from_sq) > FILE(to_sq)+1) || (FILE(from_sq) < FILE(to_sq)-1)))
      { play.b.type = CASTLE; }

  } else {    // if it is a castle

    if (!strcmp(mstring, "O-O") || !strcmp(mstring, "0-0")
        || !strcmp(mstring, "o-o") || !strcmp(mstring, "O-O+")
        || !strcmp(mstring, "0-0+") || !strcmp(mstring, "o-o+")) {
      if (wtm) { play.b.from = plist[WHITE][KING][1]; play.b.to = 6; play.b.type = CASTLE; }
      else { play.b.from = plist[BLACK][KING][1]; play.b.to = 62; play.b.type = CASTLE; }
    } else if (!strcmp(mstring, "O-O-O") || !strcmp(mstring, "0-0-0")
               || !strcmp(mstring, "o-o-o") || !strcmp(mstring, "O-O-O+")
               || !strcmp(mstring, "0-0-0+") || !strcmp(mstring, "o-o-o+")) {
      if (wtm) { play.b.from = plist[WHITE][KING][1]; play.b.to = 2; play.b.type = CASTLE; }
      else { play.b.from = plist[BLACK][KING][1]; play.b.to = 58; play.b.type = CASTLE; }
    }
  }

  //cout << "Here! " << int(play.b.from) << " " << int(play.b.to) << " " << int(play.b.type) << "\n";

  // Match up the move in the move list
  for (int z = 0; z < list.count; z++)
  {
   if (match_count > 2) break;
   if (list.mv[z].m.b.to == play.b.to &&
        (promote == list.mv[z].m.b.promote || !list.mv[z].m.b.promote))
    {
     if (list.mv[z].m.b.from == play.b.from) {
       // skip rare cases of ambiguous castling in Chess960
       if(play.b.type != CASTLE && list.mv[z].m.b.type == CASTLE) continue;
       if(play.b.type == CASTLE && list.mv[z].m.b.type != CASTLE) continue;
       mplay[match_count] = list.mv[z].m;
       match_count++;
     } else if (FILE(list.mv[z].m.b.from) == from_file &&
		PTYPE(sq[list.mv[z].m.b.from]) == piece) {
       mplay[match_count] = list.mv[z].m;
       match_count++;
     } else if (RANK(list.mv[z].m.b.from) == from_row &&
		PTYPE(sq[list.mv[z].m.b.from]) == piece) {
       mplay[match_count] = list.mv[z].m;
       match_count++;
     } else if (from_sq == -1 && from_file == -1 && from_row == -1 &&
		PTYPE(sq[list.mv[z].m.b.from]) == piece) {
       // skip rare cases of ambiguous castling in Chess960
       if(play.b.type != CASTLE && list.mv[z].m.b.type == CASTLE) continue;
       if(play.b.type == CASTLE && list.mv[z].m.b.type != CASTLE) continue;
       mplay[match_count] = list.mv[z].m;
       match_count++;
     }
    }
  }

  // check the case where there are two matches but one might
  // lead to check so is illegal... the other is then a valid move
  while (match_count <= 2 && match_count) {
    t_pos = (*this);
    legal = t_pos.exec_move(mplay[0], 0);   // tenatively making the move...
    if(match_count == 1) break;
    t_pos = (*this);
    if(legal && t_pos.exec_move(mplay[1], 0))
      { legal = 0; break; }
    else if(legal) break;
    t_pos = (*this);
    legal = t_pos.exec_move(mplay[1], 0);
    mplay[0] = mplay[1];
    break;
  }

  // if it is not legal
  if (!legal || !match_count) {
    return nomove;
  }

  return mplay[0];
}

//----------------------------------------------------------
// Function to print a single move to a string in 
// long algebraic format. This function works by simply 
// adding the appropriate characters to the move string
//----------------------------------------------------------
void position::print_move(move pmove, char mstring[10], ts_thread_data *temps)
{
  char dummy[10];             // dummy character string
  int ptype, pfrom, pto, ppiece;

  strcpy(mstring, "");
  strcpy(dummy, "");

  if(pmove.b.type&CASTLE) {           // if it is a castle
    if(FILE(pmove.b.to) == 2) snprintf(mstring, 10, "O-O-O");
    else snprintf(mstring, 10, "O-O");
  } else {
    ppiece = PTYPE(sq[pmove.b.from]);
    pfrom = pmove.b.from;
    pto = pmove.b.to;
    ptype = pmove.b.type;

//  if(!ics) {
      if (ppiece > PAWN)
       { snprintf(mstring, 10, "%c", name[ppiece]); }
      switch(ppiece) {
       case PAWN:
         if(ptype&CAPTURE)
          snprintf(dummy, sizeof(dummy), "%c", char(FILE(pfrom)+97));
         break;
       case KING:
         strcpy(dummy, ""); break;
       default:
         move_list list;
         int add_file = 0; 

         // generate the legal moves
         allmoves(&list, temps);
         // Match up the move in the move list
         for (int z = 0; z < list.count; z++) {
          if (list.mv[z].m.b.to == pto && list.mv[z].m.b.from != pfrom &&
	      ppiece == PTYPE(sq[list.mv[z].m.b.from])) {
            if(!add_file) {
              snprintf(dummy, sizeof(dummy), "%c", char(FILE(pfrom)+97));
              add_file = 1;
            }
            if(FILE(pfrom) == FILE(list.mv[z].m.b.from)) { // || ics) {
              char dummy2[4];
              snprintf(dummy2, sizeof(dummy2), "%i", (RANK(pfrom)+1));
              strcat(dummy,dummy2);
              z = list.count;
            }             
          }        
         }        
      }
      if(ptype&CAPTURE) strcat(dummy,"x");      
//   } else {  // If this is ics mode
//      sprintf(dummy,"%c%i", char(FILE(pfrom)+97),(RANK(pfrom)+1));
//   }
    strcat(mstring, dummy);
    snprintf(dummy, sizeof(dummy), "%c%i", char(FILE(pmove.b.to)+97), (RANK(pmove.b.to)+1));
    strcat(mstring, dummy);
    if (pmove.b.type&PROMOTE) {
      strcat(mstring, "="); snprintf(dummy, sizeof(dummy), "%c", name[pmove.b.promote]);
      strcat(mstring, dummy);
    }
  }

  // now execute the move and see if it leads to check or check-mate...
  position temp_pos = (*this);
  temp_pos.exec_move(pmove, 0); 
  if(temp_pos.in_check()) { // && !ics) {
    if(temp_pos.in_check_mate()) strcat(mstring, "#");
    else strcat(mstring, "+");
  }
}

// ----------------------------------------------- 
// function to write FEN notation of a position
// -----------------------------------------------
void position::write_fen(int trailing_cr) {

  char outstring[400];
  int lcount; square lsqr;

  for(int rank = 7; rank >= 0; rank--) {
    lcount = 0;
    for(int file = 0; file < 8; file++) {
      lsqr = sq[SQR(file,rank)];
      if(PTYPE(lsqr)) {
        if(lcount) {
 	 snprintf(outstring, sizeof(outstring), "%i", lcount);
 	 write_out(outstring);
 	 lcount = 0;
        }
        if(PSIDE(lsqr)) {
	  snprintf(outstring, sizeof(outstring), "%c", name[PTYPE(lsqr)]);
	  write_out(outstring);
        } else {
	  snprintf(outstring, sizeof(outstring), "%c", bname[PTYPE(lsqr)]);
	  write_out(outstring);
        }
      } else lcount ++;
    }
    if(lcount) {
      snprintf(outstring, sizeof(outstring), "%i", lcount); 
      write_out(outstring); 
      lcount = 0; 
    }
    if(rank) write_out("/");
  } 

  if(wtm) write_out(" w ");
  else write_out(" b ");

  if(castle&1) write_out("K");
  if(castle&2) write_out("Q");
  if(castle&4) write_out("k");
  if(castle&8) write_out("q");
  if(!castle)  write_out("-");

  if(ep) {
    snprintf(outstring, sizeof(outstring), " %c%i", char(FILE(ep) + 97), (RANK(ep) + 1));
    write_out(outstring);
  } else write_out(" -");

  if(trailing_cr) write_out("\n");

}

// ----------------------------------------------- 
// Function to log search output to a file 
// ----------------------------------------------- 
void tree_search::log_search(int score)
{
  char outstring[400], mstring[10];
  position p = tdata[0].n[0].pos;
  
  int total_time = GetTime()-start_time;
  unsigned __int64 node_count = 0ULL;
  for(int ti=0; ti<THREADS; ti++) { node_count += tdata[ti].node_count; }

  //---------------------------------------------------
  // Put max_ply and score information into outstring
  //---------------------------------------------------
  if(tdata[0].fail == 1) {
    snprintf(outstring, sizeof(outstring), "++\n");
    write_out(outstring);
    if(post && xboard) {
      cout << outstring;
      cout.flush();
    }
    return;
  } else if(tdata[0].fail == -1) {
    snprintf(outstring, sizeof(outstring), "--\n");
    write_out(outstring);
    if(post && xboard) {
      cout << outstring;
      cout.flush();
    }
    return;
  } else {
   snprintf(outstring, sizeof(outstring), "  %2i  %5i ", max_ply, int(float(score*100)/value[PAWN]));
  }

  //---------------------------------------------------
  // Add time and node count
  //---------------------------------------------------
  { size_t _len = strlen(outstring); snprintf(outstring + _len, sizeof(outstring) - _len, " %7i %6llu ", int(total_time), node_count); }

  //---------------------------------------------------
  // Add the PV
  //---------------------------------------------------
  int j = 0;
  if(!p.wtm && tdata[0].pc[0][0].t) {
    { size_t _len = strlen(outstring); snprintf(outstring + _len, sizeof(outstring) - _len, " 1. ..."); }
    j++;
  }

  for (int i = 0; i < MAXD && j < MAXD; i++, j++) {
    if (!(tdata[0].pc[0][i].t) || strlen(outstring) > 380) break;
    p.print_move(tdata[0].pc[0][i], mstring, &(this->tdata[0]));
    if(!p.exec_move(tdata[0].pc[0][i],0)) break;
    if(!(j&1)) {
      size_t _len = strlen(outstring); snprintf(outstring + _len, sizeof(outstring) - _len, " %i. %s", (j/2+1), mstring);
    } else {
      size_t _len = strlen(outstring); snprintf(outstring + _len, sizeof(outstring) - _len, " %s", mstring);
    }
  }
  { size_t _len = strlen(outstring); snprintf(outstring + _len, sizeof(outstring) - _len, "\n"); }

  // Write to Logfile
  write_out(outstring); 
  if(total_time >= 300) logfile.flush();

  // Do a search display if we are posting
  if(post && xboard) { 
    cout << outstring;
    cout.flush();
  }

}


// -------------------------------------------------
// Function to write search output 
//   -- similar to above but output goes to cout
//    *and* info recorded for a running testsuite
//   -- Only for use in text interface
// ------------------------------------------------- 
void tree_search::search_display(int score)
{
 char outstring[400], mstring[10];
 position p = tdata[0].n[0].pos;
 int new_line = 0;

 unsigned __int64 node_count = 0ULL;
 for(int ti=0; ti<THREADS; ti++) { node_count += tdata[ti].node_count; }

 int total_time = GetTime()-start_time;

 // if we have changed root moves, add new line
 //  -- The last_displayed_move variable carries this information
 //     It is set to zero if we also changed moves on the previous iteration
 //     and so already inserted a new line at the end of that line.
 if((analysis_mode && last_displayed_move.t) || 
    (last_displayed_move.t && last_displayed_move.t != tdata[0].pc[0][0].t && tdata[0].fail != -1)) { 
   cout << "\n"; 
   new_line = 1;
 }
 
 if(tdata[0].fail == 1 && !new_line)  { cout << " <++> "; cout.flush(); return;  }
 if(tdata[0].fail == 1 && new_line) { cout << setw(3) << max_ply << ".   ++   ";  }
 if(tdata[0].fail == -1)              { cout << " <--> "; cout.flush(); return;  }
 
 // clear current line -- 80 character width is assumed
 if(tdata[0].fail == 0 && !new_line) { 
   cout << "\r                                                                              \r";
   cout.flush();
 }

 if(score < MATE/2 && score > -MATE/2) {
   snprintf(outstring, sizeof(outstring), " %2i.  %5.2f ", max_ply, float(score)/value[PAWN]);
 } else if (score >= MATE/2) {
   snprintf(outstring, sizeof(outstring), " %2i.  MATE+%i", max_ply, +MATE-score);
 } else {
   snprintf(outstring, sizeof(outstring), " %2i.  MATE-%i", max_ply, -MATE-score);
 }

 if(!tdata[0].fail) cout << outstring;
 snprintf(outstring, sizeof(outstring), " %4i %8llu  ", int(total_time)/100, node_count);
 cout << outstring;

 int j = 0;
 if(!p.wtm && tdata[0].pc[0][0].t) {
  cout << " 1. ...";
  j++;
 }

 for (int i = 0; i < MAXD && j < MAXD; i++, j++)
 {
   if (!(tdata[0].pc[0][i].t)) break;
   p.print_move(tdata[0].pc[0][i], mstring, &(this->tdata[0]));
   if(!p.exec_move(tdata[0].pc[0][i],0)) break;
   if(!(j&1)) cout << " " << (j/2 + 1) << ".";
   cout << " " << mstring;
   // truncate long lines
   if(!xboard && !analysis_mode && i == 3 && tdata[0].pc[0][i+1].t) { 
     cout << " ... ";
     break;
   }
 }

 if(new_line) { 
   cout << "\n";
   last_displayed_move.t = 0;  // to keep track that we inserted a new line already
 } else {
   last_displayed_move.t = tdata[0].pc[0][0].t;
 }

 cout.flush();

 if(tsuite) {
  int correct = 0;
  for(unsigned int e = 0; e < bmcount; e++) {
   if(tdata[0].pc[0][0].t == bmoves[e].t && bmtype[0] == 'b') { correct = 1; break; }
   if(tdata[0].pc[0][0].t == bmoves[e].t && bmtype[0] == 'a') { break; }
   if(e == bmcount-1 && bmtype[0] == 'a') { correct = 1; }
  }
  if(!correct) soltime = -1;
  if(correct && soltime == -1) soltime = float(total_time)/100;
  best_score = score;
 }

}
