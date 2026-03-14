// search.h header file for search.cpp

#ifndef SEARCH_H
#define SEARCH_H

#define TIME_FLAG 123456

// Null move flags
int NULL_MOVE = 1;

int EGTB = 0;                       // maximum tablebase pieces

int BOOK_LEARNING = 0;          // flag for book learning

// Chess Skill
int CHESS_SKILL = 100;

// Score drop to prevent LMR in root node
int NO_ROOT_LMR_SCORE = 85;
// Score drop to extend time of search -- interact with search window!!
int EXTEND_TIME_SCORE = 15;
// Per-move overhead buffer (centiseconds) reserved when increment < 0.1s
// Accounts for interface latency and OS scheduling jitter
int MOVE_OVERHEAD_CS = 2;

// Margin for verifying null move
int VERIFY_MARGIN = 200;

// Score below alpha at which not to do qchecks
// (several other conditions apply)
#define NO_QCHECKS 650

// Draw score...
int DRAW_SCORE = -20;

// tuning parameters
//float VAR1 = 0.0915;
//float VAR2 = 7.095;
//float VAR3 = 0.1865;
float VAR1 = 4.95; //2.54;
float VAR2 = 2.46;
float VAR3 = 20.1;
float VAR4 = 2.685;



// abort search fraction
// --> tuned with following code added to search.cpp for CLOP
// float abort_search_fraction = 1.0;
// if(depth < VAR2) abort_search_fraction -= VAR1*(VAR2-depth);
// if(depth < VAR4) abort_search_fraction -= VAR3*(VAR4-depth);

// the following are divided by 1000 in calculation
int abort_search_fraction[8] = { 0, 128, 406, 625, 717, 808, 900, 1100 };

// Pruning score margins
#define MARGIN(x)  (50+100*(x)*(x))

#if DEBUG_SEARCH
 ofstream search_outfile;
#endif

int CHECK_INTER=4095;  // node interval to check time

#define SEARCH_INTERRUPT_CHECK() \
  if(tdata->ID == 0 && !(tdata->node_count&CHECK_INTER)) { \
   if (ts->max_ply > 3 || ts->ponder) {		   \
     int elapsed = GetTime() - ts->start_time;	   \
     if ((elapsed >= MIN(2*ts->limit,ts->max_limit) && !ts->ponder)	\
         || (ts->tsuite && elapsed >= ts->limit) \
	 || (inter())) {			 \
       return -TIME_FLAG;			 \
     }						 \
     if(FLTK_GUI && ts->ponder			\
	&& ts->root_wtm == gr->pos.wtm) {       \
       return -TIME_FLAG;			\
     }                                          \
   }                                            \
 }                                              \
 if(tdata->done) {                              \
   return -TIME_FLAG;                           \
 }						\


#endif  /* SEARCH_H */









