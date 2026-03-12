/* Pre-processor definitions to make the code more
   readable and easy to modify */

#ifndef DEFINE_H
#define DEFINE_H

// -------------------------------
// Program Version 
// -------------------------------
#ifdef VERS
 #define VERS2 ""
#else
 #define VERS  "7.97b"              // program version number
 #define VERS2 ""
#endif


// -------------------------------
// Some compiler specific defines
// -------------------------------
// Compiler flags for different systems
#define MSVC            0   // Selects a win95/NT compiler if set to 1
                            //   this used to work with older versions of MSVC
                            //   haven't compiled with one in several years
#ifndef MINGW
#define MINGW           0   // Set to 1 for GCC compiler in Windows
#endif
#define UNIX            1   // Set to 1 for all Unix systems
#define DEBUG_SEARCH    0   // Set to 1 to debug mode... quite slow

#ifndef MAX_THREADS
#define MAX_THREADS    32   // max number of threads possible
#endif

// Use MTD(f) algorithm with a granular eval
#ifndef USE_MTD
#define USE_MTD         0   // set to 1 to use MTD(f), otherwise PVS is used
#endif
#define EVAL_GRAN       8

// Turn off asserts by default, to turn on compile with DEBUG_ASSERT defined
// on the command line... e.g.  g++ -o EXchess ../src/EXchess.cc -O2 -D DEBUG_ASSERTS=1
#ifndef DEBUG_ASSERTS
 #define NDEBUG
 #define TEST_ASSERT
#endif

// define a mode to allow scaled depth limit with game stage
#ifndef TRAINING_MODE
 #define TRAINING_MODE   0
#endif

// NNUE evaluation (Stockfish HalfKAv2-hm compatible)
// Compile with -D NNUE=1 and place the net file next to the binary.
#ifndef NNUE
 #define NNUE 0
#endif

// Default .nnue file to load.  Override at compile time:
//   perl comp.pl MYVER NNUE=1 NNUE_NET=nn-newnet.nnue
// The companion .tdleaf.bin name is derived from NNUE_NET automatically.
#ifndef NNUE_NET
 #define NNUE_NET "nn-ad9b42354671.nnue"
#endif
#ifndef NNUE_TDLEAF_BIN
 #define NNUE_TDLEAF_BIN "nn-ad9b42354671.tdleaf.bin"
#endif

// Return material-only score from score_pos(), bypassing all positional eval.
// Compile with -D MATERIAL_ONLY=1.
#ifndef MATERIAL_ONLY
 #define MATERIAL_ONLY 0
#endif

// TDLeaf(λ) online learning for the NNUE FC layers.
// Requires NNUE=1.  Compile with -D NNUE=1 -D TDLEAF=1.
#ifndef TDLEAF
 #define TDLEAF 0
#endif

// TDLEAF_READONLY: load FC weights from .tdleaf.bin at startup (to use
// previously trained weights) but skip all ply recording and weight updates.
// Compile with -D NNUE=1 -D TDLEAF=1 -D TDLEAF_READONLY=1.
#ifndef TDLEAF_READONLY
 #define TDLEAF_READONLY 0
#endif

// Epoch-based replay: after each game, run TDLEAF_REPLAY_K additional passes
// over the last TDLEAF_REPLAY_BUF_N completed games (ring buffer).
// Score_stm is refreshed from stored accumulators against current weights each
// pass.  Set TDLEAF_REPLAY_K=0 to disable; preserves exact current behaviour.
#ifndef TDLEAF_REPLAY_K
 #define TDLEAF_REPLAY_K    1   // replay passes per game (0 = disabled)
#endif
#ifndef TDLEAF_REPLAY_BUF_N
 #define TDLEAF_REPLAY_BUF_N 8  // ring buffer capacity (~4.5 MB × N static BSS)
#endif

// define 64 bit integers and zero values for unsigned long long
#if MSVC 
 #define ZERO 0ui64
#else
 #define __int64 long long
 #define ZERO 0ULL
#endif

// require max_unit to be 64 bit so it can be combined with
// hash codes for indexing various hash tables.
#define MAX_UINT 4294967295ULL

// IO flags
#if MSVC
 #define IOS_IN_TEXT std::ios::in
#else 
 #define IOS_IN_TEXT ios::in
#endif

#define IOS_OUT ios::out | ios::binary
#define IOS_IN  ios::in | ios::binary

// -------------------------------
//  Tablebases
// -------------------------------
#ifndef TABLEBASES
 #define TABLEBASES   0  // Set to 1 to enable tablebase code, see
#endif                   // the file 'probe.cpp' for more details

// -------------------------------
//  FLTK GUI switch
// -------------------------------
#if MAKE_GUI
 #define FLTK_GUI 1      // compile the FLTK GUI interface
#else
 #define FLTK_GUI 0
#endif

// -------------------------------
//  General purpose defines below
// -------------------------------

#define MAX_MAIN_TREE  80        // maximum depth in main tree, 127 or less (due to hash depth storage limits)
#define MAXD  100                // max search depth (qsearch + main tree)
#define MAXT  3600000            // max search time in seconds (1000 hours default)
#define MATE 10000000            // mate score

// Maximum number of book positions in a temp file
#define BOOK_POS 200000
// Maximum number of temp files
#define TEMP_FILES 250

// book learning threshold..
#define LEARN_SCORE    500
#define LEARN_FACTOR     5

// maximum game length
#define MAX_GAME_PLY   1000
// maximum number of moves in position
#define MAX_MOVES      220

// Color flags
#define WHITE 1
#define BLACK 0

/* Piece definitions */

#define EMPTY        0
#define PAWN         1
#define KNIGHT       2
#define BISHOP       3
#define ROOK         4
#define QUEEN        5
#define KING         6

/* Piece id numbers for certain kinds of indicies */
// NOTE: (2012_06_13) shift in White piece values by 2 to accomodate
//  new (simpler) character based square values, see 
//  chess.h -- this requires a new HASH_ID macro to 
//  appropriately assign the correct hash values for 
//  those piece-square combinations.  See PTYPE, PSIDE
//  ID and HASH_ID macros below.
#define BPAWN        1
#define BKNIGHT      2
#define BBISHOP      3
#define BROOK        4
#define BQUEEN       5
#define BKING        6
#define WPAWN        9
#define WKNIGHT     10
#define WBISHOP     11
#define WROOK       12
#define WQUEEN      13
#define WKING       14

/* Types of moves */

#define CAPTURE      1
#define CASTLE       2
#define EP           4
#define PAWN_PUSH    8
#define PAWN_PUSH7  16
#define PROMOTE     32
#define SINGULAR    64
#define NOT_SINGULAR 63
//#define SPECIAL    128
//#define NOT_SPECIAL 127

/* macros */
#define RANK(x)    ((x)>>3)                    // Find the rank associated with square x
#define FILE(x)    ((x)&7)                     // Find the file associated with square x
#define SQR(x,y)   ((x)+((y)<<3))              // Find square from file x and rank y
#define COLOR(x)   ((RANK(x)&1)^(FILE(x)&1))   // Find the color of a square, light = 1, dark = 0 

#define CHAR_FILE(x) (int(x)-97)  // convert letter character to a file
#define CHAR_ROW(x)  (int(x)-49)  // convert number character to a row

// find the piece type and side on a square
#define PTYPE(x) ((x)&7)
#define PSIDE(x) ((x)>>3)

// find the id number of the piece
//  -- now just the value of the square, but HASH_ID is needed
//     to extract the right piece-square hash value (range 0-12)
#define ID(x) (x)   
#define HASH_ID(x)  ((PSIDE(x) > 0) ? (PTYPE(x)+6) : (PTYPE(x)) )

#define MAX(x,y) (((x) > (y)) ? (x) : (y) )
#define MIN(x,y) (((x) < (y)) ? (x) : (y) )

#define ABS(x) (((x) < 0) ? (-(x)) : (x))

#define NOMOVE       0           // no move

#endif /* DEFINE_H */













