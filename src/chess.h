/* General definition file for structures */

#ifndef CHESS_H
#define CHESS_H

// move structure to encapsulate several important move parameters
// note: The char used below is treated like an integer.

struct move_t {
  uint8_t from;                   // from square
  uint8_t to;                     // to square
  uint8_t type;                   // type of move (defined below)
  uint8_t promote;                // type of piece to promote to (defined below)
};

/*   Type of move  */
//     1 = capture
//     2 = castle
//     4 = en passant
//     8 = 2 square pawn advance (could leave an en passant square)
//    16 = pawn push 
//    32 = promotion
//    64 = singular

/*   Type of piece to promote to */
//     2 = Knight
//     3 = Bishop   
//     4 = Rook
//     5 = Queen 

// union of move_t and an integer to make comparison of 
// moves easier.  (as suggested in Tom Kerrigans simple chess program)

union move {
  move_t b;
  int32_t t;           // assuming a 32 bit integer
};

// Add a score for sorting purposes to the move record

struct move_rec {
  move m;
  int score; 
};

// Define a move_list structure to carry these move records

struct move_list {
  int count;
  move_rec mv[MAX_MOVES];

  move_list() {
    count = 0;
    for(int i = 0; i < MAX_MOVES; i++) {
      mv[i].m.t = NOMOVE;
      mv[i].score = 0;
    }
  }
};

// Typedef for square using unsigned 8 bits...  
//  -- first 3 bits for piece type (0-6)
//  -- next bit for side of piece (1 = white, 0 = black)
// See define.h for marcos and definitions to handle these

typedef uint8_t square;

// Structure for hash code and key of a position

typedef uint64_t h_code;

#include "hash.h"

#if NNUE
#include "nnue.h"
// Convenience macros for passing the NNUE accumulator through score_pos call sites.
//   NNUE_ACC_PARAM : add the optional parameter to the score_pos declaration
//   NNUE_ACC_ARG   : pass the current search_node's accumulator (use inside search_node methods)
//   NNUE_ACC_NULL  : pass nullptr  (for call sites without a search_node accumulator)
#define NNUE_ACC_PARAM , NNUEAccumulator *nnue_acc = nullptr  // declaration (has default)
#define NNUE_ACC_DEF   , NNUEAccumulator *nnue_acc            // definition  (no default)
#define NNUE_ACC_ARG   , &acc
#define NNUE_ACC_NULL  , nullptr
#else
#define NNUE_ACC_PARAM
#define NNUE_ACC_DEF
#define NNUE_ACC_ARG
#define NNUE_ACC_NULL
#endif

struct ts_thread_data;
struct tree_search;
struct game_rec;

// Structure for current board position.
//  256 bytes in size   
struct position {
  square sq[64];             // array of board squares
  int8_t wtm;                // flag for white to move
  int8_t castle;             // castling status
  int8_t ep;                 // location of an en-passant square (if any)
  int8_t fifty;              // fifty move count
  int8_t check;              // is the side to move in check?
  int8_t pieces[2];          // # of pieces(non-pawns)
  int material;              // material value from point of view of
                             // side to move

  unsigned char threat_square;
  unsigned char threat_check;

  move hmove;                // expected best move
  move rmove;                // reply move               
  move cmove;                // combination move
 
  int8_t gstage;             // game stage, range 0-16 -- depends only  
                             //    upon pawn structure --> important for pawn hashing!
                             // NOTE: **must** be integer so it is subtracted properly
                             //   -- should check this for all subtractions!  unsigned ints
                             //      may be converted incorrectly if they are negated.
  int8_t plist[2][7][10];    // piece lists
  move last;                 // last move made
  h_code hcode;              // hash code
  h_code pcode;              // pawn hash code

  int8_t qchecks[2];         // do checks in qsearch?

  int8_t Krook[2];           // castling rook squares... generalized for Chess960
  int8_t Qrook[2];

  /* initialize everything just to be sure */
  position() {
    for(int i = 0; i < 64; i++) { sq[i] = 0; }
    wtm = 0;
    castle = 0;
    ep = 0;
    fifty = 0;
    check = 0;
    pieces[0] = 0; pieces[1] = 0;
    threat_square = 0;
    threat_check = 0;
    hmove.t = 0;
    rmove.t = 0;
    cmove.t = 0;
    gstage = 0;
    for(int i = 0; i < 2; i++) {
      for(int j = 0; j < 7; j++) {
	for(int k = 0; k < 10; k++) {
	  plist[i][j][k] = 0;
	}
      }
    }
    last.t = 0;
    hcode = 0ULL;
    pcode = 0ULL;
    qchecks[0] = 0; qchecks[1] = 0;
    Krook[0] = 0; Krook[1] = 0;
    Qrook[0] = 0; Qrook[1] = 0;
  };

  /* moves.cpp */
  void allmoves(move_list *list, ts_thread_data *tdata);
  int verify_move(move_list *list, ts_thread_data *tdata, move tmove);
  void add_move(int fsq, int tsq, move_list *list, char type, ts_thread_data *tdata);
  void pawn_moves(move_list *list, int sqr, ts_thread_data *tdata);
  void king_moves(move_list *list, int sqr, ts_thread_data *tdata);
  void knight_moves(move_list *list, int sqr, ts_thread_data *tdata);
  void bishop_moves(move_list *list, int sqr, ts_thread_data *tdata);
  void rook_moves(move_list *list, int sqr, ts_thread_data *tdata);

  /* captures.cpp */
  int captures(move_list *list, int delta_score);
  int add_capt(int fsq, int tsq, move_list *list, char type, int delta_score);
  int pawn_capts(move_list *list, int sqr, int ds);
  int king_capts(move_list *list, int sqr, int ds);
  int knight_capts(move_list *list, int sqr, int ds);
  int bishop_capts(move_list *list, int sqr, int ds);
  int rook_capts(move_list *list, int sqr, int ds);

  /* captchecks.cpp */
  void captchecks(move_list *list, int delta_score);
  void add_cc(int fsq, int tsq, move_list *list, char type, int delta_score);
  void pawn_cc(move_list *list, int sqr, int ds);
  void king_cc(move_list *list, int sqr, int ds);
  void knight_cc(move_list *list, int sqr, int ds);
  void bishop_cc(move_list *list, int sqr, int ds);
  void rook_cc(move_list *list, int sqr, int ds);

  /* exmove.cpp */
  int exec_move(move emove, int ply);

  /* attacks.cpp */
  int simple_check(int move_sq);
  int dia_slide_attack(int A, int B);
  int dia_slide_attack_xray(int A, int B);
  int hor_slide_attack(int A, int B);
  int hor_slide_attack_xray(int A, int B);
  int attacked(int sqr, int side);

  /* score.cpp */
  //void init_score(int T, tree_search *ts);
  int score_pos(game_rec *gr, ts_thread_data *tdata NNUE_ACC_PARAM);
  int score_pawns(pawn_data *pawn_record);
  int score_king(game_rec *gr, int *wtropism, int *btropism);
  int bishop_mobility(int sqr, int ksq, int *kattacks, pawn_data *pawn_record);
  int rook_mobility(int sqr, int ksq, int *kattacks, pawn_data *pawn_record);
  int knight_mobility(int sqr, int ksq, int *kattacks, pawn_data *pawn_record);
  int pawn_guard(int tsq, int side, pawn_data *pawn_record);

  /* check.cpp */
  int in_check();
  int in_check_mate();

  /* hash.cpp */
  void gen_code();

  /* support.cpp */
  void write_fen(int trailing_cr);
  void print_move(move pmove, char mstring[10], ts_thread_data *temps);
  move parse_move(char mstring[10], ts_thread_data *temps);

};

// structure for nodes in the search

struct search_node {
  position pos;
#if NNUE
  NNUEAccumulator acc;   // NNUE accumulator for this node (parallel to pos)
#endif
  move_list moves;
  int best;          // best score in node
  int save_alpha;    // original alpha of node
  int null_hash;     // null hash switch
  int score;         // score returned by search, depth_mod;
  int mcount;        // which move are we searching
  int first;         // are we still searching the first move 
  int depth_mod;     // modification to depth for search
  int premove_score; // score of position before a move is made
  int pv_node;       // Is this a pv node?

  int singular;

  int hscore, hflag, hdepth;  // hash parameters

  h_code ckey;       // variables for combination hash move 
  cmove_rec *cmove;

  move smove;        // current move we are searching

  int ply;           

  // Variables for work sharing 
  int mi, skipped; unsigned char skip_move[24];

  ts_thread_data *tdata;  // parent tree search thread data
  tree_search *ts;        // parent tree search
  game_rec *gr;           // parent game record

  search_node *prev;
  search_node *next;

  // Initialize everything, just to be sure...
  search_node() {
    // pos and moves initialized when created 
    best = 0;
    save_alpha = 0;
    null_hash = 0;
    score = 0;
    mcount = 0;
    first = 0;
    depth_mod = 0;
    premove_score = 0;
    pv_node = 0;
    singular = 0;
    hscore = 0;
    hflag = 0;
    hdepth = 0;
    ckey = 0ULL;
    smove.t = 0;
    ply = 0;
    mi = 0;
    skipped = 0;
    for(int i = 0; i < 24; i++) { skip_move[i] = 0; }
    // note that various pointers are initialized when
    //  the tree search data is initialized
  };

  /* search.cpp */ 
  void root_pvs();
  int pvs(int alpha, int beta, int depth, int in_pv, int skip_move);
  int qsearch(int alpha, int beta, int qply);

};


// structures for search data
//  -- tree_search and ts_thread_data which 
//     contains thread specific data
//

struct ts_thread_data {

  unsigned int ID;         // identity of thread
  int alpha;               // initial alpha passed to thread
  int beta;                // initial beta passed to thread
  int depth;               // initial depth passed to thread   
  int g;                   // score returned from thread
  int done;                // flag to terminate search
  int running;             // flag to indicate thread is running
  int quit_thread;         // flag to tell thread to quit

  search_node n[MAXD+1];   // array of search positions
  move pc[MAXD+1][MAXD+1]; // triangular array for search
                           //    principle continuation
  h_code plist[MAX_GAME_PLY];   // hash codes of positions visited

  int fail;                     // fail high(+1)/low(-1) flag  
  int tb_hit;                   // was there a tablebase hit in search? 1/0 = y/n

  int killer1[2], killer2[2], killer3[2]; // killer moves

  // Format of these move tables is piece_id, to-square
  int history[15][64];          // table for history scores
  int reply[15][64];            // table of reply (counter) moves

  // These are a collection of counters that keep track of search
  // statistics and timing checks.
  unsigned __int64 node_count, eval_count, extensions, qchecks;
  unsigned __int64 phash_count, hash_count, hmove_count, q_count;
  unsigned __int64 null_cutoff, internal_iter, egtb_probes, egtb_hits;
  unsigned __int64 fail_high, first_fail_high, shash_count, sing_count;
 
  // function to initialize data for this thread
  void init_thread_data(int ponder) {
    // setting counts to zero
    node_count = 0ULL; eval_count = 0ULL; shash_count = 0ULL;
    phash_count = 0ULL; hash_count = 0ULL; hmove_count = 0ULL; q_count = 0ULL;
    null_cutoff = 0ULL; extensions = 0ULL; qchecks = 0ULL; internal_iter = 0ULL;
    egtb_probes = 0ULL; egtb_hits = 0ULL; fail_high = 0ULL; first_fail_high = 0ULL;
    sing_count = 0ULL;

    // if we are not pondering, age history and initialize reply table
    if(!ponder) {
      for(int i = 0; i < 15; i++)
	for(int j = 0; j < 64; j++) { history[i][j] /= 2; reply[i][j] = 0; }
    }

    // initialize triangular pv array -- except first move
    for(int pi = 0; pi < MAXD+1; pi++)
      for(int pj = pi; pj < MAXD+1; pj++)
	if(pi+pj) pc[pi][pj].t = NOMOVE;

    // initialize killer moves
    killer1[0]=0; killer1[1]=0;
    killer2[0]=0; killer2[1]=0;
    killer3[0]=0; killer3[1]=0;
  }

  /* search.cpp */
  inline void pc_update(move pcmove, int ply);

};


//--------------------------------------------
// Structure for SMP search work sharing...
//--------------------------------------------
struct share_work_data {
  h_code hcode_compare;  // will be or'd with move, depth, alpha, beta, for
                         // comparison to see if a given move is being searched

  inline int check_active_move(h_code h, int d, int a, int b, int m) {
    if(hcode_compare == (h^(h_code(m)|(h_code((d+(a^b)))<<32)))) {
      return 1;
    } else {
      return 0;
    }
  }

  inline void set_active_move(h_code h, int d, int a, int b, int m) {
    hcode_compare = h^(h_code(m)|(h_code((d+(a^b)))<<32));
  }

};

struct tree_search {

  ts_thread_data *tdata;   // data for given thread of search

  int max_ply;             // max ply (depth) of current search
  int start_time;          // start time of current search
  int limit;               // nominal time limit of search
  int max_limit;           // absolute time limit of search
  int time_double;         // number of times limit has doubled in search
  int ponder;              // flag for pondering
  int last_ponder;         // flag for did we ponder last move?
  int ponder_time;         // record of time used on last pondering
  int ponder_time_double;       // time doublings required (pondering)
  int wbest, wply;              // whisper variables for search summary
  int turn;                     // Current game turn
  int root_alpha, root_beta;    // values at the root of search 
  int root_tb_score, root_wtm;
  int start_depth;              // start depth of search  
  int last_depth;               // depth of previous search
  int g_last;                   // last returned score of search 
  int h_id;                     // flag for hash id (identifies current search)
  int max_search_depth;         // search depth limit set by xboard   
  int max_search_time;          // search time limit set by xboard  
  int fail_high;                // flag for failing high
  int fail_low;                 // flag for failing low 

  move bookm;                   // move from opening book
  move ponder_move;             // move we are pondering
  move last_displayed_move;     // best move from previous search display

  int tsuite, analysis_mode;    // flags to determine whether we are in
                                //    analysis mode or a test suite

  move_list root_moves;         // movelist for the root position

  move singular_response;       // store a singular response

  // threads for SMP
  int thread_index;                              
  pthread_mutex_t runlock[MAX_THREADS];
  pthread_mutex_t waitlock[MAX_THREADS];
  pthread_mutex_t waitlock2[MAX_THREADS];
  pthread_cond_t go_cond[MAX_THREADS];
  pthread_t search_thread[MAX_THREADS];
  int initialized_threads;
  

  // share data for SMP search
  share_work_data share_data[MAXD+1][MAX_THREADS];

  // variables to support testsuite mode 
  float soltime;
  unsigned int bmcount; move bmoves[256];
  int best_score;
  char bmtype[3];      // "am" avoid move or "bm" best move string                              

  int best_depth;               // best depth reached (for testsuites) 
  int no_book;                  // counter for positions with no book moves

  /* tree_search_functions.cpp */
  tree_search();           // initializer function 
  ~tree_search();          // destructor function 
  void initialize_extra_threads();
  void close_extra_threads();
  void delete_thread_data();
  void create_thread_data(game_rec *gr, int thread_count);
  void history_stats();

  /* search.cpp */
  move search(position p, int time_limit, int T, game_rec *gr);
  int search_threads(int alpha, int beta, int depth, int threads);
  void sort_root_moves();

  /* support.cpp */
  void search_display(int score);
  void log_search(int score);
   
};


// Structure for position in the opening book

struct book_rec {
 uint64_t pos_code;    // position hash code
 uint16_t score;       // score for position
 uint16_t gambit;      // flag for gambit play
 uint16_t wins;        // number of wins with this pos 
 uint16_t losses;      // number of losses with this pos, also
                               //  works as extra padding to book_rec to guarantee it falls on
                               //  word boundaries on 32 and 64 bit systems 
};

// Structure to hold all the data and flags for a game

struct game_rec {
 /* game positions */
  position pos;            // current position in the game
  position last;           // previous position in the game
  position temp;           // temporary position for book-keeping
  position reset;          // reset position for takebacks
 /* available moves */
  move_list movelist;      // list of pseudo-legal moves for current pos
  move best;               // best move for current position
 /* game history info */
  move game_history[MAX_GAME_PLY];  // list of move played

 /* game control flags and counters */
  int T;                   // turn number
  int p_side;              // side taken by opponent
  int book;                // flag about whether to use book
  int both;                // opponent plays both sides (0 = false)
  int mttc;                // moves 'til time control
  int omttc;               // opponents moves 'til time control
  int searching;           // flag for when we are searching
  int force_mode;          // flag to indicate when we are in force mode (xboard)
  int process_move;        // flag if there is a move to process
  int terminate_search;    // flag for terminating search on interruption
  int program_run;         // flag to indicate EXchess is running (hasn't been told to quit)
  float inc;               // time increment with each move (seconds)
  int over;                // game is over
  char overstring[40];     // game over message
  char lmove[10];          // string with last played move
  int learned;             // has book learning already happened?
  int learn_count;         // count of book moves to learn
  int learn_bk;            // flag to control book learning
  float base;              // base amount of time (seconds)
  float timeleft[2];       // total time left (centi-seconds)
  int knowledge_scale;     // variable to weaken EXchess (up to 100 for full strength)

 /* search structure for searches */ 
  tree_search ts;

  /* functions in game_rec.cpp */
  game_rec();
  void setboard(const char inboard[256], const char ms, const char castle[5], const char ep[3]);
  void test_suite(char *testfile, char *resfile, float testtime, int fixed_depth);
  void board_edit();
  void build_fen_list();
  void train_eval();
  
}; 

#endif  /* CHESS_H */









