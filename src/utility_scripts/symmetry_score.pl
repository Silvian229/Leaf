#!/bin/perl
#
# script to make a symmetric version of piece square tables
# from a score.par file
# 
# usage:  perl symmetry_score.pl infile outfile
# 

$ifile = $ARGV[0];
$ofile = $ARGV[1];

open(IN,"$ifile") || die "Cannot open $ifile\n";
open(OUT, ">$ofile") || die "Cannot open $ofile\n";

$modify_line = 0;

while(<IN>) {
    $line = $_;
    if(!$modify_line) { print OUT $line; }
    if($line =~ /PIECE_TAB:/) {
	$modify_line = 1;
    } elsif ($modify_line) {
	@vals = split(' ');
	$vals[0] = int(($vals[0]+$vals[7])/2.0);
	$vals[7] = $vals[0];
	$vals[1] = int(($vals[1]+$vals[6])/2.0);
	$vals[6] = $vals[1];
	$vals[2] = int(($vals[2]+$vals[5])/2.0);
	$vals[5] = $vals[2];
	$vals[3] = int(($vals[3]+$vals[4])/2.0);
	$vals[4] = $vals[3];
	printf OUT "%4i %4i %4i %4i %4i %4i %4i %4i\n", 
	       $vals[0], $vals[1], $vals[2], $vals[3],
	       $vals[4], $vals[5], $vals[6], $vals[7];
	$modify_line++;
	if($modify_line == 9) { $modify_line = 0; }
    }
}
