#
# Perl script to take score.par file and output piece-sq tables in
#   format compatitabile with score.h file.  
#

$output_line = 0;

$stage = 0; $piece = 1; $square = 0;

@pname = ( "NO_PIECE", "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING" );

open(IN, "score.par") || die "Cannot open score.par!\n";

while(<IN>) {
    $line = $_;
    #if(!$output_line) { print $line; }
    if($line =~ /PIECE_TAB:/) {
        $output_line = 1;
    } elsif ($output_line) {
        @vals = split(' ');
	for($i = 0; $i < 8; $i++) {
	    $pvals[$stage][$piece][$square] = $vals[$i]/10;
	    $square++;
	}
        $output_line++;
        if($output_line == 9) { 
	    $output_line = 0; 
	    $stage++;
	    $square = 0;
	    if($stage > 3) { $piece++; $stage = 0; }
	}
    }
}


#
# Output
#

for($stage = 0; $stage < 4; $stage++) {
    print "{\n";
    for($piece = 0; $piece < 7; $piece++) {
	print "//For $pname[$piece] in stage $stage\n";
	print "{\n";
	for($square = 0; $square < 64; $square++) {
	    $sqr_val = 0;
	    if($piece > 0) { $sqr_val = $pvals[$stage][$piece][$square]; }
	    if(!($square%8)) { print "      "; }
	    printf "%4i", $sqr_val;
	    if($square < 63) { print ","; }
	    if(($square%8) == 7) { print "\n"; }
	}
	print "}";
	if($piece == 6) { print " }"; }
	if($stage < 3 || $piece < 6) { print ",";}
	print "\n";
    }
}

print "};\n";
