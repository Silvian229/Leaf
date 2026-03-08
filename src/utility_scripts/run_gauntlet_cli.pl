#
# Script to run a testmatch against a suite of 'standard' engines
#  using cutechess-cli.
#
# Set parameters of matches below
#
#
$LAST_MODIFIED = "2013_05_17";
#
# Change Log
#   2013_05_17 -- removing the cpu cooling period because now running
#                  with turbo mode disabled
#   2013_01_30 -- turning off updates to dropbox
#   2013_01_24 -- updating for current version of cutechess-cli
#   2013_01_22 -- removing some opponents and changing tc
#   2012_10_22 -- changes opponents tc 
#   2012_08_07 -- new opponents and tc odds added
#   2012_05_12 -- new opponents and random number handling
#   2012_03_10 -- modified for new linux system
#   2012_01_06 -- modified to use engines.json file
#   2012_01_05 -- modified to work with cutechess_cli
#   2012_01_02 -- modified output of bayeselo
#   2011_12_31 -- modified output of bayeselo
#   2011_06_06 -- more fixes, arasanx removed for now
#   2011_06_05 -- modified to use somewhat longer time controls
#                  and changed reference EXchess version
#                  Also testing Arasanx as an opponent
#   2011_05_02 -- modified to do much faster time controls
#   2010_08_21 -- modified time odds for phalanx, crafy, and arasanx
#   2010_08_16 -- changed learning strategy a bit to change position
#                  every time
#   2010_08_15 -- include possibility for time odds
#   2010_08_14 -- added ability to do learning matches
#                 added init string to turn off book for EXchess
#   2010_08_11 -- added debug statement to xboard
#                  and code to rotate debug logs
#   2010_08_08 -- script created
#

# match games to run in a round = twice number of positions in a round
$mg = 20;
# number of rounds to run  -- $mg * $rounds <= twice number of positions in file
$rounds = 75;
# book file created with
# polyglot make-book -pgn normbk02.pgn -bin normbk02.bin -max-ply 30 -min-game 5
$open_book = "normbk02.bin";
# concurrent sessions
$concur = 4;
# time control for tested engine
$tc = "tc=0/6.0+0.10";  

#--------------------------------------------------------
# Names refer to engines defined in 'engines.json' file
#--------------------------------------------------------
@Engine = (
#"\"Critter 1.6a\"    tc=0/0.6+0.01", 
"\"DoubleCheck 3.4\" tc=0/6.0+0.10", 
#"\"Stockfish 2.2.2\" tc=0/1.8+0.03", 
#"\"Glaurung 2.2\"    tc=0/2.4+0.04", 
#"\"Spike 1.2\"       tc=0/6.0+0.20", 
"\"Toga II 1.4.1SE\" tc=0/6.0+0.10", 
"\"Crafty-23.4\"     tc=0/6.0+0.10", 
"\"Gaviota 0.85\"    tc=0/6.0+0.10", 
"\"GNU Chess 6.0.1\" tc=0/6.0+0.10", 
"\"Arasan 14.3\"     tc=0/6.0+0.10"  
);

#----------------------------------------------------------------------------------
# Get info about the engine to be tested
#----------------------------------------------------------------------------------
print "We are doing a gauntlet run...\n";
print "WARNING:  Only EXchess versions dated on or later than 2010_08_11 will work!\n\n";
print "Enter version for the engine: ";
$input1 = <STDIN>;
chomp($input1);
$test_engine = "cmd=\"./EXchess_v$input1 xb\" dir=../testvers/v$input1 name=\"EXchess v$input1\" proto=xboard $tc";
print "Using the following cutechess-cli command for engine: -engine $test_engine\n";
$sgf = "v$input1" . ".cli_gaunt3.pgn";

print "Putting the results in the file: $sgf\n";

# check if game file exists
if(-e $sgf) {
    print "File exists, append results or delete [A/d]? ";
    $input3 = <STDIN>;
    chomp($input3);
    if($input3 eq 'd') { 
	print "Deleting existing version...\n"; 
	unlink($sgf);
    } else {
	print "Appending results to existing file.\n";
    }
}
	
#----------------------------------------------------------------------------------
# Use a temporary positions file to only do a small number of games in each round
#
# Now asking about number of rounds and starting position from big epd file
#----------------------------------------------------------------------------------
print "Enter number of rounds [default = 75]: ";
$input2 = <STDIN>;
chomp($input2);
if($input2) { $rounds = $input2; } 
print "Enter number of concurrent CPUs [default = 4]: ";
$input2 = <STDIN>;
chomp($input2);
if($input2) { $concur = $input2; } 
print "Using $rounds rounds of $mg games each.  Games are distributed over $concur CPUs\n"; 

#----------------------------------------------------------------------------------
# Setup the cutechess commands
#----------------------------------------------------------------------------------
$options = "-games $mg -concurrency $concur -repeat -draw movenumber=160 movecount=4 score=100 -recover";
$engine_options = "book=$open_book bookdepth=8";
$cutechess_command = "/usr/local/bin/cutechess-cli";

print "Are these settings correct (y/n)? ";
$response = <STDIN>;
chomp($response);
if(!($response =~ /y/ || $response =~ /Y/)) {
    print "Please run script again and enter new values.\n";
    print "Quitting now...\n"; 
    exit; 
} 

print "Would you like to compare results against an earlier version? (y/n): ";
$response = <STDIN>;
chomp($response);
if(($response =~ /y/ || $response =~ /Y/)) {
    $compare_earlier = 1;
    print "Enter comparison PGN file: ";
    $compare_file = <STDIN>;
    chomp($compare_file);
} else {
    $compare_earlier = 0;
}

#----------------------------------------------------------------------------------
# Now loop over rounds and engines to run a gauntlet
#----------------------------------------------------------------------------------
for($j = 0; $j < $rounds; $j++) {
    #if($j) { print "Sleeping for 60 seconds (CPU cooling)\n"; sleep(60); }
    for($engine_i = 0; $engine_i < 6; $engine_i++) {
	print "--------------------------------------------------------------------------------\n";
	printf "Matching v$input1 vs $Engine[$engine_i] in round %i of $rounds\n", ($j+1);
	$first_engine = "conf=$Engine[$engine_i]";
	$seed = int(rand(2147483648));
	$command = "$cutechess_command -engine $first_engine -engine $test_engine -each $engine_options -srand $seed $options -pgnout $sgf";
	print "Executing $command\n";
	system("$command\n");
	print "Deleting log and learning files (if any) and rotating xboard.debug\n";
	system("rm -rf game.* log.* *.lrn\n");
	print "--------------------------------------------------------------------------------\n";
    }
    #----------------------------------------------------------------------------------
    # Use the bayeselo program to print a summary so far + put it in the Dropbox folder
    #----------------------------------------------------------------------------------
    open(OUTELO, ">elo.script");
    print OUTELO "prompt off\n";
    if($compare_earlier) { 
	print OUTELO "readpgn $compare_file\n";
    }
    print OUTELO "readpgn $sgf\n";
    print OUTELO "elo\n";
    print OUTELO "mm\n";
    print OUTELO "exactdist\n";
    print OUTELO "offset 0 EXchess v$input1\n";
    print OUTELO "ratings\n";
    print OUTELO "los\n";
    print OUTELO "details\n";
    close(OUTELO);
    system("~/chessprograms/BayesElo/bayeselo < elo.script > v$input1.cli_gaunt3_ratings.txt\n");
    system("cat v$input1.cli_gaunt3_ratings.txt\n");
#    system("cp v$input1.cli_gaunt3_ratings.txt ~/Dropbox/curr_results.txt\n");
    print "\n--------------------------------------------------------------------------------\n";
}

    
exit;

