#
# Script to run a testmatch against a suite of 'standard' engines
#  using cutechess-cli.
#
# Set parameters of matches below
#
#
$LAST_MODIFIED = "2013_01_30";
#
# Change Log
#   2013_01_30 -- turning off updates to dropbox
#   2013_01_24 -- updated for new cutechess-cli
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
$rounds = 30;
# book file created with
# polyglot make-book -pgn normbk02.pgn -bin normbk02.bin -max-ply 30 -uniform
$open_book = "normbk02.bin";
# time control in minutes -- will be divided by at least 10 
$tc = 6;  
# increment in seconds -- will be divided by at least 10
$inc = 0.1;
# concurrent sessions
$concur = 4;

#----------------------------------------------------------------------------------
# Get info about the engine versions to be tested
#----------------------------------------------------------------------------------
print "We are doing a selftest...\n";
print "WARNING:  Only EXchess versions dated on or later than 2010_08_11 will work!\n\n";
print "Enter test version for the engine: ";
$input1 = <STDIN>;
chomp($input1);
$test_engine1 = "cmd=\"./EXchess_v$input1 xb\" dir=../testvers/v$input1 name=\"EXchess v$input1\" proto=xboard";
print "Using the following cutechess-cli command for engine: -engine $test_engine1\n";
$sgf = "v$input1" . ".cli_selftest.pgn";
print "Enter reference version for the engine: ";
$input1b = <STDIN>;
chomp($input1b);
$test_engine2 = "cmd=\"./EXchess_v$input1b xb\" dir=../testvers/v$input1b name=\"EXchess v$input1b\" proto=xboard";
print "Using the following cutechess-cli command for engine: -engine $test_engine2\n";

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
print "Enter number of rounds [default = 30]: ";
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
$engine_options = "tc=0/$tc+$inc book=$open_book";
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
# Now loop over rounds and engines to run a selftest
#----------------------------------------------------------------------------------
for($j = 0; $j < $rounds; $j++) {
    print "--------------------------------------------------------------------------------\n";
    printf "Matching v$input1 vs v$input1b in round %i of $rounds\n", ($j+1);
    $first_engine = "conf=$Engine[$engine_i]";
    $command = "$cutechess_command -engine $test_engine2 -engine $test_engine1 -each $engine_options $options -pgnout $sgf";
    print "Executing $command\n";
    system("$command\n");
    print "Deleting log and learning files (if any) and rotating xboard.debug\n";
    system("rm -rf game.* log.* *.lrn\n");
    print "--------------------------------------------------------------------------------\n";
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
    system("~/chessprograms/BayesElo/bayeselo < elo.script > v$input1.cli_selftest_ratings.txt\n");
    system("cat v$input1.cli_selftest_ratings.txt\n");
#    system("cp v$input1.cli_selftest_ratings.txt ~/Dropbox/curr_self_results.txt\n");
    print "\n--------------------------------------------------------------------------------\n";
}

    
exit;

