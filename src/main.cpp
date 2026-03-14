// EXchess source code, (c) Daniel C. Homan  1997-2017
// Released under the GNU public license, see file license.txt

// Thanks to Jim Ablett for fixes to the polling code which 
// allow EXchess to work with the mingw compiler on windows!

/*  Main functions controlling program */
//-------------------------------------------
// int main(int argc, char *argv[])
// void takeback(int tm)
// void make_move()
// void drawboard()
// void help()
// void type_moves()
// void type_capts()
// void parse_command()
// void performance()
// void save_game()
// int inter()
// void write_out(const char *outline) {
//-------------------------------------------

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <fstream>
#include <stdint.h>
#include <pthread.h>

#include "define.h"
#include "stdtypes.h"
#include "chess.h"
#include "search.h"

// MUST come after define.h to disable asserts
#include <assert.h>

#if MSVC || MINGW
 #include <windows.h>
 #include <time.h>
 #include <conio.h>
 #if MSVC
  #undef BLACK
  #undef WHITE
  #define BLACK 0
  #define WHITE 1
 #endif
#else 
 #include <time.h>
 #include <unistd.h>
 #include <sys/types.h>
 #include <sys/time.h>
#endif

// Custom headers, defining external functions and struct types for
// board, piece and moves.  And defining global variables.

#include "const.h"
#include "funct.h"
#include "hash.h"
#include "extern.h"

// Main Game Structure
game_rec game;

// Logging variables
ofstream logfile;
int MAX_LOGS = 100;

// Flags for input and display
int display_board = 0;
char response[60];        // first word of input string from command prompt

// set of values for tracking interface lag
#define LAG_COUNT  10
int interface_lags[LAG_COUNT];
int interface_lag_count = 0;
int next_lag_mod = 0;
int average_lag = 0;

// Basic control flags for whole game
int xboard = 0, post = 0, ics = 0, ALLEG = 0, hintflag = 0;
int ponder_flag = 1, shout_book = 0;

// Executable directory
char exec_path[FILENAME_MAX];

#if UNIX
 fd_set read_fds;
 struct timeval timeout = { 0, 0 };
#endif

#if FLTK_GUI
 int FLTKmain(int argc, char** argv);
#endif

/*------------------------- Main Function ---------------------------*/
//  Main control function that interacts with the User

int main(int argc, char *argv[])
{
  char mstring[10];
  move hint;

#ifdef TEST_ASSERT
  assert(0);  // testing that asserts are off
#endif

  //-------------------------------
  // Parse and record exec path
  //-------------------------------
  strcpy(exec_path, argv[0]);
  int last_slash = -1;
  for(int j = 0; j < FILENAME_MAX; j++) {
    if(exec_path[j] == '\0') break;
    if(exec_path[j] == '\\') last_slash = j;
    if(exec_path[j] == '/') last_slash = j;
  }  
  if(last_slash == 0) strcpy(exec_path, "./");
  else exec_path[last_slash+1] = '\0';

  // initialize lock for logfile and egtb...
  // -- must come before any function that might
  //    try to access these locks, even "write_out"
  pthread_mutex_init(&log_lock,NULL);
  pthread_mutex_init(&egtb_lock,NULL);

  //---------------------------------
  // Initialize variables for 
  //  -- search params 
  //  -- hash table
  //  -- check tables
  //  -- random number seeds
  //  -- tablebases
  //  -- threads and locks
  //  -- board and game record
  //----------------------------------
  set_search_param();
  set_hash_size(HASH_SIZE);
  gen_check_table();
  srand(time(NULL));
#if TABLEBASES
  init_tb();
#endif

#if NNUE
  {
    // --init-nnue: create a fresh random-initialised .nnue without reading an existing one.
    // Requires --write-nnue <filename>; writes the file and exits.
    bool init_nnue_mode = false;
    for (int ai = 1; ai < argc; ai++)
      if (strcmp(argv[ai], "--init-nnue") == 0) { init_nnue_mode = true; break; }

    if (init_nnue_mode) {
#if !TDLEAF
      fprintf(stderr, "--init-nnue requires a TDLEAF build (compile with -D TDLEAF=1)\n");
      return 1;
#else
      // Verify --write-nnue is also present.
      bool has_write = false;
      for (int ai = 1; ai < argc - 1; ai++)
        if (strcmp(argv[ai], "--write-nnue") == 0) { has_write = true; break; }
      if (!has_write) {
        fprintf(stderr, "--init-nnue requires --write-nnue <filename>\n");
        return 1;
      }
      // Allocate arrays, init float shadows, then fill with random distributions.
      nnue_alloc_arrays();
      nnue_init_fp32_weights();
      nnue_init_zero_weights();
      // Fall through to the --write-nnue handler below, which will write and exit.
#endif
    } else {
      char nnue_path[FILENAME_MAX];
      snprintf(nnue_path, sizeof(nnue_path), "%s%s", exec_path, NNUE_NET);
      if (!nnue_load(nnue_path)) {
        nnue_load(NNUE_NET);
      }
      if (nnue_available) write_out("NNUE evaluation loaded.\n");
      else                write_out("NNUE file not found, using classical evaluation.\n");
#if TDLEAF
      if (nnue_available) {
        // Attempt to load previously learned weights (companion .tdleaf.bin file).
        char tdleaf_path[512];
        snprintf(tdleaf_path, sizeof(tdleaf_path), "%s%s",
                 exec_path, NNUE_TDLEAF_BIN);
        bool loaded = nnue_load_fc_weights(tdleaf_path);
        if (!loaded)
            loaded = nnue_load_fc_weights(NNUE_TDLEAF_BIN);
        if (!loaded)
          fprintf(stderr, "TDLeaf: no weights file found — using pretrained .nnue weights.\n"
                          "TDLeaf: run with --init-nnue --write-nnue <file> to create a fresh net.\n");
      }
#endif
    }
  }
#endif

#if NNUE
  // Handle --write-nnue <filename>: export current NNUE weights (FC trained via TDLeaf,
  // FT copied verbatim from the loaded .nnue) to a complete .nnue file, then exit.
  for (int ai = 1; ai < argc; ai++) {
    if (strcmp(argv[ai], "--write-nnue") == 0 && ai + 1 < argc) {
      if (!nnue_available) {
        fprintf(stderr, "--write-nnue: no NNUE loaded\n");
        return 1;
      }
      if (!nnue_write_nnue(argv[ai + 1])) {
        fprintf(stderr, "--write-nnue: failed to write %s\n", argv[ai + 1]);
        return 1;
      }
      return 0;
    }
  }
#endif

  // initialize all thread data
  game.ts.create_thread_data(&game, MAX_THREADS);
  game.ts.initialize_extra_threads();

  // initialize board and movelist
  game.setboard(i_pos, 'w', "KQkq", "-");

  //-------------------------------
  // opening logging file 
  //-------------------------------
  if(logging) {
    char lfile[FILENAME_MAX];
    strcpy(lfile, exec_path);
    strcpy(lfile, "run.log");
    for(int li = 1; li <= ABS(MAX_LOGS); li++) {
      // if there is a single log file (MAX_LOGS = 1), 
      // just append to that or create one if none is there
      if(li==1 && MAX_LOGS == 1) {
	logfile.open(lfile, ios::in); 
	if(logfile.is_open()) { 
	  logfile.close(); 
	  logfile.open(lfile, ios::app); 
	  break;
	} else { 
	  logfile.open(lfile, ios::out);
	  break;
	}
      }
      // if there is a single log file with MAX_LOGS = -1, 
      // overwrite the exisiting log that is there
      if(li==1 && MAX_LOGS == -1) {
	logfile.open(lfile, ios::out);
	break;	
      }
      // otherwise find a new file name for log up to MAX_LOGS
      if(li < 10) snprintf(lfile, sizeof(lfile), "%srun_00%i.log", exec_path, li);
      else if(li < 100) snprintf(lfile, sizeof(lfile), "%srun_0%i.log", exec_path, li);
      else snprintf(lfile, sizeof(lfile), "%srun_%i.log", exec_path, li);
      logfile.open(lfile, ios::in);
      if(!logfile.is_open()) {
	logfile.close();
	logfile.open(lfile, ios::out);       
	break;
      } else {
	logfile.close();
      }
    }
    if(!logfile.is_open()) {
      cout << "Error(Can't open logging file!)\n";
      logging = 0;
    } else {
      logfile.clear();
      logfile << "===============================\n"; 
      logfile << "=\n";
      logfile << "= Log file for Leaf v" << VERS << VERS2 << "\n";
      logfile << "=\n";
      logfile << "===============================\n"; 
    }
  }

  //-----------------------------------
  // parsing command line args
  //  -- virtually no error checking!
  //-----------------------------------
  for(int argi = 1; argi < argc; argi++) {
    // turn on xboard mode
    if(!strcmp(argv[argi], "xb")) { 
      xboard = 1;
      continue;
    }
    // set the number of cores to use
    if(!strcmp(argv[argi], "cores")) { 
      int thread_val = atoi(argv[argi+1]);
      if(thread_val > MAX_THREADS) {
	cout << "Error(MAX_THREADS set to " << MAX_THREADS << " so cores limited to this value)\n";
	logfile << "Error(MAX_THREADS set to " << MAX_THREADS << " so cores limited to this value)\n";
        THREADS = MAX_THREADS;
	game.ts.initialize_extra_threads();
      } else if(thread_val < 1) {
	cout << "Error(THREADS must be at least 1, no change made)\n";
	logfile << "Error(THREADS must be at least 1, no change made)\n";
      } else {
	THREADS = thread_val;
	game.ts.initialize_extra_threads();
      }
      argi += 1;
      continue;
    }
    // set hash size in MB
    // FORMAT:  hash <size_in_MB>   
    if(!strcmp(argv[argi], "hash")) { 
      HASH_SIZE = ABS(atoi(argv[argi+1]));
      set_hash_size(HASH_SIZE);
      argi += 1;
      continue;
    }
    // command line score value change, use centipawn as smallest unit
    // FORMAT:  setvalue <parameter_name> <parameter_value>
    if(!strcmp(argv[argi], "setvalue")) { 
      set_score_value(argv[argi+1], atof(argv[argi+2]));
      argi += 2; 
      continue;
    }
    // test command has to be last on command line, will exit after
    // FORMAT:  test <epd_file> <results_file> <time_in_seconds> <depth_limit>
    if(!strcmp(argv[argi], "test")) {
      game.test_suite(argv[argi+1], argv[argi+2], atof(argv[argi+3]), atoi(argv[argi+4]));
      close_hash();
      return 0;
    }
  }

#if FLTK_GUI
#if UNIX
  /* throw away cout output */
  std::streambuf *nullbuf;
  ofstream nullstr;
  nullstr.open("/dev/null");
  nullbuf = nullstr.rdbuf();
  cout.rdbuf(nullbuf);
#endif
  /* Start main loop for GUI interface */
  return FLTKmain(argc,argv);
#endif

  if(!xboard) {
    cout << "\nLeaf Chess Engine version " << VERS << VERS2 << " (beta),"
         << "\nCopyright (C) 1997-2026 Daniel C. Homan, Granville OH, USA"
         << "\nLeaf comes with ABSOLUTELY NO WARRANTY. This is free"
         << "\nsoftware, and you are welcome to redistribute it under"
         << "\ncertain conditions. This program is distributed under the"
         << "\nGNU public license.  See the file license.txt"
         << "\nfor more information.\n\n";

    //    cout << sizeof(cmove_rec) << "\n";

    cout << "Hash size = " << TAB_SIZE << " buckets of 4 entries, "
         << TAB_SIZE*sizeof(hash_bucket)/1048576 << " Mbytes\n";
    cout << "Pawn size = " << PAWN_SIZE << " individual entries, "
         << PAWN_SIZE*sizeof(pawn_rec)/1048576 << " Mbytes\n";
    cout << "Score size = " << SCORE_SIZE << " individual entries, "
         << SCORE_SIZE*sizeof(score_rec)/1048576 << " Mbytes\n";
    cout << "Cmove size = " << CMOVE_SIZE << " individual entries, "
         << CMOVE_SIZE*sizeof(cmove_rec)/1048576 << " Mbytes\n\n";
    cout << "Type 'help' for a list of commands.\n";
    if(MATERIAL_ONLY) {
        cout << "\nWARNING -- Material-only eval mode active.\n";
    }
  } else { 

    // catch signals for xboard interface
    signal(SIGINT, SIG_IGN);

  }



  /* main loop for text interface */

  while (game.program_run)
   {

    // find a hint move, check book first then look in pv
    if(hintflag) {
      hint.t = 0;
      if(game.ts.last_ponder) hint = game.ts.ponder_move;
      else if(game.book) hint = opening_book(game.pos.hcode, game.pos, &game);
      if(!hint.t) hint = game.ts.tdata[0].pc[0][1];
      if(hint.t) {
	game.pos.print_move(hint, mstring, &game.ts.tdata[0]);
       cout << "Hint: " << mstring << "\n";
      }
      hintflag = 0;
    }

    // pondering if possible
    if(game.T > 2 && game.p_side == game.pos.wtm && !game.over
       && !game.both && !game.ts.last_ponder && !game.force_mode && ponder_flag)
    {
      if(!xboard) cout << "pondering... (press any key to interrupt)\n";
      cout.flush();
      game.ts.ponder = 1;
      game.ts.search(game.pos, 1, game.T+1, &game);
      game.ts.ponder = 0;
      game.ts.last_ponder = 1;
    }

    // if analysis_mode, do some analysis
    if(game.ts.analysis_mode && !game.over && !game.force_mode) {
      game.p_side = !game.pos.wtm;
      game.ts.search(game.pos, 36000000, game.T, &game);
      game.p_side = game.pos.wtm;
    }

    // if we received a quit command while pondering or analyzing
    if(!game.program_run) break;

    if(!game.pos.wtm)                        // if it is black's turn
    {
     if(game.both) game.p_side = 0;
     if(!xboard) cout << "Black-To-Move[" << floor((double)game.T/2) << "]: ";
     if(logging) logfile << "Black-To-Move[" << floor((double)game.T/2) << "]: ";
    }
    else                                         // or if it is white's
    {
     if(game.both) game.p_side = 1;
     if(!xboard) cout << "White-To-Move[" << (floor((double)game.T/2) + 1) << "]: ";
     if(logging) logfile << "White-To-Move[" << (floor((double)game.T/2) + 1) << "]: ";
    }

    cout.flush();

    // process any move that happened while during pondering or analysis
    if(game.process_move) { make_move(); game.T++; game.process_move = 0; continue; }

    if(game.p_side == game.pos.wtm || game.over || game.force_mode) {
      cin >> response;      // get the command
      if((game.ts.last_ponder || game.ts.analysis_mode) && UNIX) cout << "\n";       
      parse_command();      // parse it
    } else {
      if(!xboard) cout << "Thinking ...\n";
      if(logging) logfile << "Thinking ...\n";
      cout.flush();
      make_move();
      game.ts.last_ponder = 0;
      game.T++;
    }

    cout.flush();
    if(logging) logfile.flush();
   }

  if(logging) logfile.close();
  close_hash();

  return 0;
}

// Function to takeback moves
// tm is the number of moves to take back.
// 1 or 2 with current setup
void takeback(int tm)
{
 int temp_turn = game.T;
 // no book learning yet
 game.learn_count = 0; game.learned = 0;
 // game is not over
 game.over = 0;
 game.pos = game.reset;
 game.T = temp_turn; if(!(game.T % 2)) game.p_side = 0;
 if(game.p_side == 0 && tm == 1) game.p_side = 1;
 for (int ip = 0; ip <= game.T-2-tm; ip++)
 {
  game.pos.exec_move(game.game_history[ip], 0);
 }
 if(!xboard && !FLTK_GUI) drawboard();
 game.T = game.T - tm;
 // find quasi-legal moves
 game.pos.allmoves(&game.movelist, &game.ts.tdata[0]);     
}


// Function to make the next move... If it is the computer's turn, this
// function calls the search algorithm, takes the best move given by that
// search, and makes the move - unless it is a check move: then it flagges
// stale-mate.....
// The function also looks to see if this move places the opponent in check
// or check-mate.

void make_move()
{
   int mtime = GetTime(); int time_limit, legal, ri;
   char mstring[10]; int time_div = 73; int rep_count = 0;
   char outstring[400];

   if (game.p_side != game.pos.wtm)
   {
     //------------------------------
     // Estimate search time to use
     //------------------------------
     if(xboard || game.mttc) {
       // Estimate the amount of time remaining to time control
       int projected_time = game.timeleft[game.pos.wtm];
       int moves_remaining = game.mttc+1;
       if(!game.mttc || moves_remaining > 40) {
	 moves_remaining = 40;
       }
       projected_time += moves_remaining*game.inc*100;
       if(interface_lag_count >= LAG_COUNT) {
         projected_time -= moves_remaining*average_lag;
       }
       // Don't let lags reduce projected time by more than half
       projected_time = MAX(projected_time, game.timeleft[game.pos.wtm]/2);
       // Base limit used on this move on the projected time
       time_limit = 75*projected_time/(100*moves_remaining);
       if(ponder_flag) { time_limit = (115*time_limit)/100; }
       // if opponent made the expected move, reduce the time...
       /*
       if(game.ts.tdata[0].pc[0][1].t == game.pos.last.t && game.ts.tdata[0].pc[0][1].t) {
	 time_limit = 100*time_limit/100;
       }
       */
       // Use no more than half of the time left
       time_limit = MIN(time_limit, game.timeleft[game.pos.wtm]/2);
       // Reserve a fixed overhead buffer for small-increment time controls
       // to account for interface latency and OS scheduling jitter per move
       if(game.inc > 0.0f && game.inc < 0.10f) {
         time_limit = MAX(1, time_limit - game.ts.MOVE_OVERHEAD_CS);
       }
     } else {
       time_limit = game.timeleft[game.pos.wtm];
     }
     //---------------------------
     // Now complete the search
     //---------------------------
     game.best = game.ts.search(game.pos, time_limit, game.T, &game);
     assert(!game.searching);
#if TDLEAF && NNUE && !TDLEAF_READONLY
     if (nnue_available) {
       tdleaf_record_ply(game.td_game,
                         game.pos,
                         game.ts.tdata[0].n[0].acc,
                         game.ts.tdata[0].pc[0],
                         game.ts.g_last,
                         (bool)game.pos.wtm,
                         game.ts.id_scores,
                         game.ts.id_score_count);
     }
#endif
     //---------------------------
     // Adjust remaining time
     //---------------------------
     game.timeleft[game.pos.wtm] -= float(GetTime() - mtime); 
     game.timeleft[game.pos.wtm] += float(game.inc*100);
   }

   // reduce time control by one move after every pair of moves in the game, should avoid the
   // winboard 'bug' introduced when forced moves are made by at the start of the game for the engine
   if(!(game.T&1)) {
    if(game.mttc) { 
      game.mttc--; 
      if(!game.mttc) { 
	game.timeleft[0] += game.base*100; 
	game.timeleft[1] += game.base*100; 
	game.mttc = game.omttc; 
      } 
    }
    if(game.mttc <= 0 && !xboard) { 
      game.timeleft[0] = game.base*100; 
      game.timeleft[1] = game.base*100;
      game.mttc = game.omttc; 
    }
   }

   // execute the move....
   game.temp = game.pos;
   legal = game.temp.exec_move(game.best, 0);

   // if game isn't over, make the move
   if(!game.over) {
    
    // Is the move legal? if not Error ....
    if (legal) {
      game.pos.print_move(game.best, mstring, &game.ts.tdata[0]);
      strcpy(game.lmove,mstring);
      // if it is the computer's turn - echo the move
      if(game.p_side != game.pos.wtm || FLTK_GUI) {
	if(!xboard) {
	  if(game.pos.wtm) {
	    cout << (ceil((double)game.T/2) + 1) << ". ";
	    snprintf(outstring, sizeof(outstring), "%i. ", (int(((double)game.T)/2) + 1));
	    write_out(outstring);
	  } else {
	    cout << ceil((double)game.T/2) << ". ... ";
	    snprintf(outstring, sizeof(outstring), "%i. ... ", int(((double)game.T)/2));
	    write_out(outstring);
	  }
	} else {
	  cout << "move ";
	}
        cout << mstring << "\n"; cout.flush();
        write_out(mstring); write_out("\n");
      }

      game.last = game.pos;        // Save last position
      game.pos = game.temp;        // actually execute move

      // Check if we have, check_mate, stale_mate, or a continuing game...
      switch (game.pos.in_check_mate()) {
       case 0:
         if(game.pos.fifty >= 100) { 
           game.over = 1;
	   cout << "1/2-1/2 {50 moves}\n";
           snprintf(game.overstring, sizeof(game.overstring), "1/2-1/2 {50 moves}");
	   write_out(game.overstring);
           if(ics) cout << "tellics draw\n"; 
	 } else if(game.pos.in_check() && !xboard) {
	   cout << "Check!\n"; 
	 }
         // check for a 3-rep
         for(ri = game.T-2; ri >= game.T-game.pos.fifty && rep_count < 2; ri -= 2)
          if(game.ts.tdata[0].plist[ri] == game.pos.hcode) {
           rep_count++;
           if(rep_count > 1) {
	     game.over = 1;
	     cout << "1/2-1/2 {3-rep}\n";
	     snprintf(game.overstring, sizeof(game.overstring), "1/2-1/2 {3-rep}");
	     write_out(game.overstring);
	     if(ics) cout << "tellics draw\n";
           }
          }    
         break;
       case 1:
         game.over = 1;
         if(!game.pos.wtm) { 
	   cout << "1-0 {White Mates}\n";
           snprintf(game.overstring, sizeof(game.overstring), "1-0 {White Mates}");
	   write_out(game.overstring);
         } else {
	   cout << "0-1 {Black Mates}\n";
           snprintf(game.overstring, sizeof(game.overstring), "0-1 {Black Mates}");
	   write_out(game.overstring);
	 }
         break;
       case 2:
         game.over = 1;
         cout << "1/2-1/2 {Stalemate}\n";
	 snprintf(game.overstring, sizeof(game.overstring), "1/2-1/2 {Stalemate}");
	 write_out(game.overstring);
      }
      // Flush stdout before TDLeaf writes to stderr; without this the result
      // line stays in the stdout pipe buffer while the stderr TDLeaf message
      // is already visible to the driving script, causing it to miss the message.
      if (game.over) cout.flush();

      game.game_history[game.T-1] = game.best; // record the move in the history list
      // update position list for all threads
      for(int ti=0; ti<MAX_THREADS;ti++) {
	game.ts.tdata[ti].plist[game.T] = game.pos.hcode;
      }
      if(!xboard && display_board) drawboard();  // draw the resulting board
    } else { 
      game.over = 1; 
      cout << "Error - please reset"; 
      snprintf(game.overstring, sizeof(game.overstring), "Error - please reset\n");
      write_out(game.overstring);
    }
   } 

   // update the quasi-legal moves in this situation
   game.pos.allmoves(&game.movelist, &game.ts.tdata[0]);

#if TDLEAF && NNUE && !TDLEAF_READONLY
   // If the game just ended via checkmate/stalemate/draw, trigger TDLeaf update.
   if (game.over && nnue_available && game.td_game.n_plies > 0) {
     float td_result = 0.5f;
     if (strstr(game.overstring, "1-0"))  td_result = 1.0f;
     else if (strstr(game.overstring, "0-1")) td_result = 0.0f;
     char tdleaf_save[FILENAME_MAX];
     snprintf(tdleaf_save, sizeof(tdleaf_save), "%s%s", exec_path, NNUE_TDLEAF_BIN);
     tdleaf_update_after_game(game.td_game, td_result, tdleaf_save);
     tdleaf_replay(game.td_game, td_result, tdleaf_save);
     game.td_game.n_plies = 0;  // prevent double-trigger
   }
#endif

}

// This function draws the graphical board in a very simple way
void drawboard()
{
  char mstring[10];     // character string to hold move

 // the following for loop steps through the board and paints each square
  for (int j = 7; j >= 0; j--)
  {
   cout << "\n  +---+---+---+---+---+---+---+---+\n" << (j+1) << " | ";
   for (int i = 0; i <= 7; i++)
   {
     if(PTYPE(game.pos.sq[SQR(i,j)]) && !PSIDE(game.pos.sq[SQR(i,j)])) 
       cout << "\b<" << name[PTYPE(game.pos.sq[SQR(i,j)])] << ">| ";
     else if(PSIDE(game.pos.sq[SQR(i,j)]) == 1) cout << name[PTYPE(game.pos.sq[SQR(i,j)])] << " | ";
    else if(!((i+j)&1)) cout << "\b:::| ";
    else cout << "  | ";
   }
   if(j==7) { if(game.pos.wtm) cout << "   White to move";
                          else cout << "   Black to move"; }
   if(j==6) {
     cout << "   castle: ";
     if(game.pos.castle&1) cout << "K";
     if(game.pos.castle&2) cout << "Q";
     if(game.pos.castle&4) cout << "k";
     if(game.pos.castle&8) cout << "q";
     if(!game.pos.castle)  cout << "-";
   }
   if(j==5 && game.pos.ep)
     cout << "   ep: " << char(FILE(game.pos.ep) + 97) << (RANK(game.pos.ep) + 1);
   if(j==4 && game.pos.last.t) {
     cout << "   last: ";
     game.last.print_move(game.pos.last, mstring, &game.ts.tdata[0]);
     cout << mstring;
    }
   if(j==3) cout << "   fifty: " << ceil((double)game.pos.fifty/2);
   if(j==2) cout << "   Computer time: " << int(game.timeleft[game.p_side^1]/100) << " seconds";
  }
   cout << "\n  +---+---+---+---+---+---+---+---+";
   cout << "\n    a   b   c   d   e   f   g   h  \n\n";
}


// Help function
void help()
{

 cout <<   "\n Commands ........ ";
 cout << "\n\n   Enter a move in standard algebraic notation,";
 cout <<   "\n      Nf3, e4, O-O, d8=Q, Bxf7, Ned7, etc....";
 cout <<   "\n      Other notation's like: g1f3, e2e4, etc... are also ok.";
 cout << "\n\n   new            -> start a new game";
 cout <<   "\n   quit           -> end Leaf";
 cout <<   "\n   save           -> save the game to a text file";
 cout <<   "\n   go             -> computer takes side on move";
 cout <<   "\n   white          -> white to move, Leaf takes black";
 cout <<   "\n   black          -> black to move, Leaf takes white";
 cout <<   "\n   book           -> toggle opening book";
 cout <<   "\n   post           -> turn on display of computer thinking";
 cout <<   "\n   nopost         -> turn off display of computer thinking";
 cout <<   "\n   setboard rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
 cout <<   "\n                  -> setup board using EPD/FEN notation";
 cout <<   "\n                     The last two fields are castling rights and";
 cout <<   "\n                      the en-passant square, if possible. If";
 cout <<   "\n                      either is not, use a '-' instead.";
 cout <<   "\n   level 40 5 0   -> set level of play:";
 cout <<   "\n                       1st number is the number of move till time control";
 cout <<   "\n                       2nd number is the base time control in minutes";
 cout <<   "\n                       3rd number is the increment in seconds";
 cout <<   "\n   takeback       -> takeback last move";
 cout <<   "\n------------ Hit \"Enter\" for remaining commands: ";
 cin.get(); cin.get();
 cout <<   "\n   hint           -> get a hint from the program";
 cout <<   "\n   testsuite      -> run a testsuite";
 cout <<   "\n   display        -> display the board";
 cout <<   "\n   nodisplay      -> turn off board display";
 cout <<   "\n   list           -> list the legal moves";
 cout <<   "\n   clist          -> list the legal captures";
 cout <<   "\n   score          -> score the current position";
 cout <<   "\n   analyze        -> enter analysis mode";
 cout <<   "\n   exit           -> exit analysis mode";
 cout <<   "\n   ponder         -> toggle pondering";
 cout <<   "\n   hash n         -> set total hash to n megabytes";
 cout <<   "\n   build          -> build a new opening book from a pgn file";
 cout <<   "\n   edit_book      -> directly edit the current opening book";

 cout << "\n\n";
}

/* Function to print the possible moves to the screen */
// Useful for debugging
void type_moves()
{
  int j = 0;   // dummy count variable to determine when to
               // send a newline character
  char mstring[10]; // character string for the move

  for(int i = 0; i < game.movelist.count; i++) {
      game.temp = game.pos;
      // if it is legal, print it!
      if(game.temp.exec_move(game.movelist.mv[i].m, 0)) {
         if(!(j%6) && j) cout << "\n";    // newline if we have printed
                                          // 6 moves on a line
         else if(j) cout << ", ";         // comma to separate moves
         game.pos.print_move(game.movelist.mv[i].m, mstring, &game.ts.tdata[0]);   // print the move
	                                                                  // to the string
         cout << mstring;
         j++;                              // increment printed moves variable
      }
  }
  cout << "\n";
}

/* Function to print out the possible captures to the screen */
// Useful for debugging
void type_capts()
{
  int j = 0;              // dummy variable for counting printed moves
  char mstring[10];       // character string to hold move
  move_list clist;        // capture list

  game.pos.captures(&clist, -10000);
  for(int i = 0; i < clist.count; i++) {
      game.temp = game.pos;
      // if it is legal, print it!
      if(game.temp.exec_move(clist.mv[i].m, 0)) {
         if(!(j%6) && j) cout << "\n";    // newline if we have printed
                                          // 6 moves on a line
         else if(j) cout << ", ";         // comma to separate moves
         game.pos.print_move(clist.mv[i].m, mstring, &game.ts.tdata[0]);         // print the move
                                                                        // to the string
         cout << mstring;
         j++;                              // increment printed moves variable
      }
  }
  cout << "\n";
}

/* Function to parse the command from the user */
// Some of these commands are xboard/winboard specific

void parse_command()
{
  char inboard[256], ms, castle[5], ep[3], basestring[12];
  char outstring[400];
  char testfile[100] = "";
  char resfile[100] = "";
  char * options;
  char line[256];
  int rating1, rating2, Mbytes, fsq, tsq, protN;
  int interface_time;
  int computed_lag;
  float min = 0.0, sec = 0.0;

  // logging initial command
  if(logging) logfile << "Parsing command: " << response << "\n";

  // default is to assume command will terminate an active search
  game.terminate_search = 1;

  // if the command is an instruction
  if(!strcmp(response, "level"))
   { cin >> game.mttc >> basestring >> game.inc;
     sscanf(basestring, "%f:%f", &min, &sec);
     game.base = min*60.0+sec;
     game.omttc = game.mttc; 
     game.timeleft[0] = game.base*100; 
     game.timeleft[1] = game.base*100; }
  else if(!strcmp(response, "time")) { 
    cin >> interface_time; 
    computed_lag = game.timeleft[game.p_side^1] - interface_time;
    game.timeleft[game.p_side^1] = interface_time;
    //-------------------------------------------------
    // Keep a running list of up to LAG_COUNT lags,
    //  --> next_lag_mod variable tells which part of
    //      this array the next lag should replace,
    //      tis allows a continuously updated list
    //  --> interface_lag_count, count of lags
    //      recorded so far, up to LAG_COUNT
    //-------------------------------------------------
    interface_lags[next_lag_mod] = computed_lag;
    next_lag_mod++;
    if(next_lag_mod == LAG_COUNT) {
      next_lag_mod = 0;
    }
    if(interface_lag_count < LAG_COUNT) {
      interface_lag_count++;
    }
    //-------------------------------------------------
    // Compute the average lag to use in time alottment
    //   -- computed only over most recent LAG_COUNT
    //   -- max and min lags are discarded in average
    //      to give a more robust result 
    //-------------------------------------------------
    average_lag = 0;
    int max_lag = -1000; int min_lag = 1000;
    for(int li = 0; li < interface_lag_count; li++) {
      average_lag += interface_lags[li];
      if(interface_lags[li] > max_lag) {
	max_lag = interface_lags[li];
      }
      if(interface_lags[li] < min_lag) {
	min_lag = interface_lags[li];
      }
    }
    if(interface_lag_count > 2) {
      average_lag -= (max_lag+min_lag);
      average_lag /= (interface_lag_count-2);
    } else {
      average_lag /= interface_lag_count;
    }
    average_lag = MAX(average_lag, 0);  // don't have a lag < 0
    //-------------------------------------------------
    // Report lag statistics to log file
    //-------------------------------------------------
    snprintf(outstring, sizeof(outstring), "Lag measured to be %3i centi-sec, running <lag> = %3i\n", computed_lag, average_lag);
    write_out(outstring);
  }
  else if(!strcmp(response, "otim")) { cin >> game.timeleft[game.p_side];}
  else if(!strcmp(response, "display") && !xboard)
   { display_board = 1; drawboard(); }
  else if(!strcmp(response, "nodisplay") && !xboard)
   { display_board = 0; }
  else if(!strcmp(response, "force")) { game.both = 1; game.force_mode = 1; }
  else if(!strcmp(response, "black"))
   { game.pos.wtm = 0; game.p_side = 0; game.both = 0;
     game.pos.gen_code();
     game.pos.in_check();  }
  else if(!strcmp(response, "white"))
   { game.pos.wtm = 1; game.p_side = 1; game.both = 0;
     game.pos.gen_code();
     game.pos.in_check();  }
  else if(!strcmp(response, "go"))
    { game.p_side = game.pos.wtm^1; game.both = 0; game.force_mode = 0; game.ts.analysis_mode = 0;  }
  else if(!strcmp(response, "playother"))
    { game.p_side = game.pos.wtm; game.both = 0; game.force_mode = 0; game.ts.analysis_mode = 0;  }
  else if(!strcmp(response, "edit")) { game.board_edit(); }
  else if(!strcmp(response, "testsuite") && !xboard) { game.test_suite(testfile,resfile,0, 0); }
  else if(!strcmp(response, "analyze"))
   { post = 1; game.learn_bk = 0; game.ts.analysis_mode = 1; game.book = 0; game.both = 1;
     if(!xboard) cout << "Analysis mode: Enter commands/moves as ready.\n\n";      
   }
  else if(!strcmp(response, "exit"))
    { cout << "Analysis mode: off\n"; game.ts.analysis_mode = 0;  }
  else if(!strcmp(response, "new"))
   { if(!game.ts.analysis_mode) { game.both = 0; game.book = 1; game.learn_bk = BOOK_LEARNING; }
     game.ts.no_book = 0;
     game.ts.max_search_depth = MAXD;
     game.ts.max_search_time = MAXT;
     game.setboard(i_pos, 'w', "KQkq", "-");
     game.force_mode = 0;
  }
  else if(!strcmp(response, "?")) { 
    game.terminate_search = 1; 
  }
  else if(!strcmp(response, ".")) { 
      game.terminate_search = 0; 
      cout << "Error (unknown command): " << response << "\n"; 
      logfile << "Error (unknown command): " << response << "\n"; 
  }
  else if(!strcmp(response, "draw")) { 
    // a draw request can/will terminate a ponder but not
    // a move search... note that if we don't terminate
    // a ponder, an immediate followup command from xboard might be
    // missed leading to a search that doesn't terminate
    if(!game.ts.ponder) { game.terminate_search = 0; } 
  }
  else if(!strcmp(response, "remove"))
    { takeback(2); }
  else if(!strcmp(response, "takeback") && !xboard)
    { takeback(2); }
  else if(!strcmp(response, "undo")) { takeback(1); }
  else if(!strcmp(response, "sd")) { cin >> game.ts.max_search_depth; }
  else if(!strcmp(response, "st")) { cin >> game.ts.max_search_time; }
  else if(!strcmp(response, "bk") || !strcmp(response, "book"))
   { if(game.book) { game.book = 0; cout << " Book off\n\n"; }
     else { game.book = 1; cout << " Book on\n\n"; } }
  else if(!strcmp(response, "hint")) { hintflag = 1; }
  else if(!strcmp(response, "edit_book") && !xboard) {
   edit_book(game.pos.hcode, &game.pos);
  }
  else if(!strcmp(response, "shout")) { shout_book = 1; }
  else if(!strcmp(response, "post")) { post = 1; }
  else if(!strcmp(response, "cores")) { 
    int thread_val;
    cin >> thread_val;
    if(thread_val > MAX_THREADS) {
      cout << "Error(MAX_THREADS set to " << MAX_THREADS << " so cores limited to this value)\n";
      logfile << "Error(MAX_THREADS set to " << MAX_THREADS << " so cores limited to this value)\n";
      THREADS = MAX_THREADS;
      game.ts.initialize_extra_threads();
    } else if(thread_val < 1) {
      cout << "Error(THREADS must be at least 1, no change made)\n";
      logfile << "Error(THREADS must be at least 1, no change made)\n";
    } else {
      THREADS = thread_val;
      game.ts.initialize_extra_threads();
    }
  }
  else if(!strcmp(response, "swap") && !xboard) { 
    cin >> fsq >> tsq;  
    cout << swap(tsq,game.pos,game.pos.wtm,fsq) << "\n"; 
  }
  else if(!strcmp(response, "xboard")) { 
    /* only do something if xboard mode is not already set */
    if(!xboard) { 
      xboard = 1;
      // catch signals for xboard interface
      signal(SIGINT, SIG_IGN);
    }
  }    
  else if(!strcmp(response, "variant")) { cin.getline(line,256); }
  else if(!strcmp(response, "protover")) {
    cin >> protN;
    if(protN > 1) {
      cout << "\n";
      cout << "feature setboard=1\n";
      cout << "feature playother=1\n";
      cout << "feature usermove=1\n"; 
      cout << "feature ics=1\n";
      cout << "feature smp=1\n";
      cout << "feature memory=1\n";
      cout << "feature variants=\"normal,nocastle,fischerandom\"\n";
      cout << "feature myname=\"Leaf v" << VERS << VERS2 << "\"\n";
      snprintf(outstring, sizeof(outstring), "feature option=\"Playing Strength -slider %i 1 100\n", game.knowledge_scale);
      cout << outstring;
      cout << "feature done=1\n";
      cout.flush();
    }
  }
  else if(!strcmp(response, "option")) { 
    cin.getline(line, 256);
    if(strstr(line, "=") != NULL) {    // Thanks to Alex Guerrero for this fix!
      options = strtok(line,"=");
      if(!strcmp(options, " Playing Strength")) {
	options = strtok(NULL,"\n");
	game.knowledge_scale = atoi(options);
      }
    }
  }
  else if(!strcmp(response, "build") && !xboard) { 
    game.setboard(i_pos, 'w', "KQkq", "-");
    build_book(game.pos); 
  }
  else if(!strcmp(response, "memory") || !strcmp(response, "hash"))
    { cin >> Mbytes; set_hash_size(ABS(Mbytes));
      if(!xboard) {
	cout << " Hash size = " << TAB_SIZE << " buckets of 4 entries, "
	     << TAB_SIZE*sizeof(hash_bucket)/1048576 << " Mbytes\n";
	cout << " Pawn size = " << PAWN_SIZE << " individual entries, "
	     << PAWN_SIZE*sizeof(pawn_rec)/1048576 << " Mbytes\n"; 
	cout << "Score size = " << SCORE_SIZE << " individual entries, "
	     << SCORE_SIZE*sizeof(score_rec)/1048576 << " Mbytes\n";
	cout << "Cmove size = " << CMOVE_SIZE << " individual entries, "
	     << CMOVE_SIZE*sizeof(cmove_rec)/1048576 << " Mbytes\n\n"; 
      }
    }
  //else if(!strcmp(response, "sum") || !strcmp(response, "ics"))
  // { ics = 1; if(!xboard) cout << " Search summary is on\n\n"; }
  else if(!strcmp(response, "ponder") && !xboard)
   { if(ponder_flag) { ponder_flag = 0; cout << " Pondering off\n\n"; }
     else { ponder_flag = 1; cout << " Pondering on\n\n"; } }
  else if(!strcmp(response, "easy")) // pondering off in xboard/winboard
   { ponder_flag = 0; }
  else if(!strcmp(response, "hard")) // pondering on in xboard/winboard
   { ponder_flag = 1; }
  else if(!strcmp(response, "list") && !xboard) {  type_moves(); }
  else if(!strcmp(response, "clist") && !xboard) { type_capts(); }
  else if(!strcmp(response, "print_psq") && !xboard) { print_psq(); }
  else if(!strcmp(response, "score") && !xboard)
    { game.p_side = game.pos.wtm^1;
#if NNUE
      if (nnue_available) {
        NNUEAccumulator tmp_acc;
        nnue_init_accumulator(tmp_acc, game.pos);
        cout << "score = " << game.pos.score_pos(&game, &game.ts.tdata[0], &tmp_acc) << "\n";
      } else
#endif
      cout << "score = " << game.pos.score_pos(&game,&game.ts.tdata[0]) << "\n";
      cout << "material = " << game.pos.material << "\n";
      game.p_side = game.pos.wtm; }
  else if(!xboard && !strcmp(response, "help")) { help(); }
  else if(!strcmp(response, "nopost")) { post = 0; }
#if TABLEBASES
  else if(!strcmp(response, "probe") && !xboard) { cout << probe_tb(&game.pos,0) << "\n"; }
#endif
  else if((!strcmp(response, "save") || !strcmp(response, "SR")) && !xboard) { save_game(); }
  else if(!strcmp(response, "quit")) { game.over = 1; game.program_run = 0;  }
  else if(!strcmp(response, "result")) {
    game.over = 1;
#if TDLEAF && NNUE && !TDLEAF_READONLY
    if (nnue_available && game.td_game.n_plies > 0) {
      // xboard sends: "result 1-0 {description}" or "0-1" or "1/2-1/2"
      char result_str[20]   = "";
      char result_desc[256] = "";
      cin >> result_str;
      cin.getline(result_desc, sizeof(result_desc));
      // Skip learning if the game ended on time — the result reflects clock
      // management, not chess quality.  Both cutechess-cli and XBoard include
      // the word "time" in the description for time-forfeit results
      // (e.g. "{Black forfeits on time}", "{White wins on time}").
      if (strstr(result_desc, "time") == nullptr) {
        float td_result = 0.5f;
        if (!strcmp(result_str, "1-0"))      td_result = 1.0f;
        else if (!strcmp(result_str, "0-1")) td_result = 0.0f;
        char tdleaf_save[FILENAME_MAX];
        snprintf(tdleaf_save, sizeof(tdleaf_save), "%s%s", exec_path, NNUE_TDLEAF_BIN);
        tdleaf_update_after_game(game.td_game, td_result, tdleaf_save);
        tdleaf_replay(game.td_game, td_result, tdleaf_save);
      }
      game.td_game.n_plies = 0;  // always reset — prevent carry-over to next game
    }
#endif
  }
  else if(!strcmp(response, "setvalue")) { 
    char par_string[50]; float par_val;
    cin >> par_string >> par_val;
    set_score_value(par_string, par_val);
  }
  else if(!strcmp(response, "getvalue")) {
    char par_string[50];
    cin >> par_string;
    cout << par_string << ": "<< get_score_value(par_string) << "\n";;
  }
  else if(!strcmp(response, "performance") && !xboard) { performance(); }
  else if(!strcmp(response, "history_stats") && !xboard) 
    { game.ts.analysis_mode = 0; game.ts.history_stats(); }
  else if(!strcmp(response, "name")) 
    { cin.getline(line, 256); if(logging) logfile << "--> Opponent: " << line << "\n"; }
  else if(!strcmp(response, "ics")) 
    { ics = 1; cin.getline(line, 256); if(logging) logfile << "--> ICS Host: " << line << "\n"; }
  else if(!strcmp(response, "rating")) 
    { cin >> rating1 >> rating2; 
      if(logging) logfile << "--> My rating  : " << rating1 << "\n"
                          << "--> Opp. rating: " << rating2 << "\n"; }
  else if(!strcmp(response, "setboard"))
    { cin >> inboard >> ms >> castle >> ep; game.setboard(inboard, ms, castle, ep);  }
  else if(!strcmp(response, "accepted")) { cin.getline(line, 256); }  // ignore accepted command
  else if(!strcmp(response, "rejected")) { cin.getline(line, 256); }  // ignore rejected command
  // if command is a move
  else if(!strcmp(response, "usermove")) { 
    cin >> response; 
    cout << "Got Usermove!\n";
    game.best = game.pos.parse_move(response, &game.ts.tdata[0]);
    if(game.best.t) { 
      if(!game.searching) { make_move(); game.T++; }
      else { 
	game.process_move = 1; 	  
      }
    } else { 
      cout << "Illegal move: " << response << "\n"; 
      logfile << "Illegal move: " << response << "\n"; 
      cin.getline(line, 256);
    }
    cout.flush();
  } 
  else { 
    game.best = game.pos.parse_move(response, &game.ts.tdata[0]);
    if(game.best.t) {    
      if(!game.searching) { make_move(); game.T++; }
      else { 
	game.process_move = 1;   
      }
    } else { 
      cout << "Error (unknown command): " << response << "\n"; 
      logfile << "Error (unknown command): " << response << "\n"; 
      cin.getline(line, 256);
    }
    cout.flush();
  }
}

// Function to run a perfomance test on generating and making move
void performance()
{
 position perf_pos = game.pos;
 move_list perf_list;
 int gen_count = 0;
 int start_time = GetTime();
 int loop = 0, perfi;

 while(1) {
  perf_pos.allmoves(&perf_list, &game.ts.tdata[0]);   
  gen_count += perf_list.count;
  loop++; if(loop > 1000) { loop = 0; if(GetTime()-start_time > 500) break; }
 }

 cout << "Generated " << gen_count << " moves in " << float(GetTime()-start_time)/100 << " seconds\n";

 loop = 0; start_time = GetTime(); gen_count = 0;

 while(1) {
  perf_pos.allmoves(&perf_list, &game.ts.tdata[0]);   
  gen_count += perf_list.count;
  for(perfi = 0; perfi < perf_list.count; perfi++) {
   game.temp = perf_pos;
   game.temp.exec_move(perf_list.mv[perfi].m, 0);
  }   
  loop++; if(loop > 1000) { loop = 0; if(GetTime()-start_time > 500) break; }
 }

 cout << "Generated/Made/Unmade " << gen_count << " moves in " << float(GetTime()-start_time)/100 << " seconds\n";
}

// Save game function to save the game to a text file
void save_game()
{
  int TURN; TURN = game.T;
  char gname[] = "lastgame.gam";
  char resp, mstring[10];
  char Event[30], White[30], Black[30], Date[30], result[30];

  cout << "\nFile Name : ";
  cin >> gname;
  cout << "Custom Header? (y/n): ";
  cin >> resp;

  if(resp == 'y' || resp == 'Y')
  {
    cout << "Event: ";  cin >> Event;
    cout << "Date: "; cin >> Date;
    cout << "White: ";  cin >> White;
    cout << "Black: ";  cin >> Black;
  } else {
    strcpy(Event, "Chess Match");
    strcpy(Date, "??.??.????");
    if (game.p_side)
     { strcpy(White, "Human"); strcpy(Black, "Leaf"); }
    else
     { strcpy(White, "Leaf"); strcpy(Black, "Human"); }
  }

  ofstream outfile(gname);

  outfile <<   "[Event: " << Event << " ]";
  outfile << "\n[Date: " << Date << " ]";
  outfile << "\n[White: " << White << " ]";
  outfile << "\n[Black: " << Black << " ]";

  // set the result string
  switch (game.pos.in_check_mate())
   {
    case 0:
     if(game.pos.fifty >= 100)
      { strcpy(result, " 1/2-1/2 {50 moves}"); }
     else strcpy(result, " adjourned");
      break;
    case 1:
     if(!game.pos.wtm) strcpy(result, " 1-0 {White Mates}");
     else strcpy(result, " 0-1 {Black Mates}");
     break;
    case 2:
     strcpy(result, " 1/2-1/2 {Stalemate}");
   }

  outfile << "\n[Result: " << result << " ]\n\n";

  // set the board up from the starting position
  game.setboard(i_pos, 'w', "KQkq", "-");

  // play through the game and record the moves in a file
  for(int i = 1; i < TURN; i++)
   {
     game.pos.print_move(game.game_history[i-1], mstring, &game.ts.tdata[0]);
     if (game.pos.wtm) outfile << (ceil((double)i/2) + 1) << ". " << mstring;
     else outfile << mstring;
     outfile << " ";
     if(!(game.T%8)) outfile << "\n";
     game.pos.exec_move(game.game_history[i-1], 0);
     game.T++;
   }

   outfile << result;

   // update quasi-legal moves
   game.pos.allmoves(&game.movelist, &game.ts.tdata[0]);     

}

#if FLTK_GUI

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
//#include <FL/Fl_Bitmap.H>
//#include <FL/fl_draw.H>
//#include <FL/Fl_Menu_Item.H>
//#include <FL/fl_ask.H>

#endif

//-----------------------------------------------------------------------------------
// Function returns a 1 if a pondering session should be interrupted
int inter()
{
#if FLTK_GUI
  Fl::check();
  if(abortflag) { abortflag = 0; return 1; } else { return 0; }
#endif

  int interrupt = 0;
 
  if(!game.ts.ponder && !game.ts.analysis_mode && !xboard) return 0;

  if(!xboard && cin.rdbuf() -> in_avail() > 1) interrupt = 1;

  if(!interrupt) {
#if MSVC || MINGW  
    static int init = 0, pipe;
    static HANDLE inh;
    DWORD dw;
    if(xboard) {     // winboard interrupt code taken from crafty
      if (!init) {
	init = 1;
	inh = GetStdHandle(STD_INPUT_HANDLE);
	pipe = !GetConsoleMode(inh, &dw);
	if (!pipe) {
	  SetConsoleMode(inh, dw & ~(ENABLE_MOUSE_INPUT|ENABLE_WINDOW_INPUT));
	  FlushConsoleInputBuffer(inh);
	  FlushConsoleInputBuffer(inh);
	}
      }
      if(pipe) {
	if(!PeekNamedPipe(inh, NULL, 0, NULL, &dw, NULL)) interrupt = 1;
	interrupt = dw;
      } else {
	GetNumberOfConsoleInputEvents(inh, &dw);
	dw <= 1 ? 0 : dw;
	interrupt = dw;
      }
    }
    if(kbhit()) interrupt = 1;

#else // unix

    FD_ZERO(&read_fds);
    FD_SET(0,&read_fds);
    timeout.tv_sec = timeout.tv_usec = 0;
    select(1,&read_fds,NULL,NULL,&timeout);
    if((game.ts.ponder || game.ts.analysis_mode || xboard) && FD_ISSET(0,&read_fds)) interrupt = 1;

#endif
  }

  if(interrupt && xboard) {
    cin >> response;
    parse_command();
    if(!game.terminate_search) {
      interrupt = 0;
    } else {
      logfile << "Search Interrupted by GUI command\n";
    }
  }

  return interrupt;
}

// ------------ function to write stuff to logfile or to search posting buffer 
//
void write_out(const char *outline) {
  pthread_mutex_lock(&log_lock);
  if(logging) logfile << outline;
#if FLTK_GUI
  if(FLTK_post) { 
   searchout_buffer->append(outline);
   searchout->move_down();
   searchout->show_insert_position();
  }
#endif
  pthread_mutex_unlock(&log_lock);
}











