use IO::Handle;

open(OUT, ">temp_hash_list") || die "Cannot open outfile!\n";
OUT->autoflush(1);

my $val_count = 1;
my @vals;
$vals[0][0] = 0;  
$vals[0][1] = 0;  

my $randval32_0, $randval32_1;
my $ham_max, $ham_max;

#
# Generate 850 hash codes
#  They are generated in 32-bit upper and 32-bit lower
#  halves and then printed to the screen as a single
#  64-bit hex number
#
while($val_count <= 850) {
    $ham_max = -1;
    $ham_min = 17;
    #
    # Don't allow any 64 bit digit where the upper
    #  or lower 32 bits having a hamming distance
    #  too small or too large from the respective
    #  32 bits on any other number generate
    #
    my $i = 0;
    while($i < $val_count) {
	$randval32_0 = rand(4294967295);
	$randval32_1 = rand(4294967295);
	
	$ham_max = -1;
	$ham_min = 17;
	$miss_count = 0;
	for($i = 0; $i < $val_count && $miss_count <= 0.9*$val_count; $i++) {
	    $ham_value = hamming($vals[$i][0], $randval32_0);
	    if($ham_value > 24 || $ham_value < 8) { last; }
	    if($ham_value > 22 || $ham_value < 10) { $miss_count++; }
	    $ham_value += hamming($vals[$i][1], $randval32_1);
	    if($ham_value > 42 || $ham_value < 22) { last; }
	}
    }
    $vals[$val_count][0] = $randval32_0;
    $vals[$val_count][1] = $randval32_1;
    $val_count++;
    $numstring = sprintf "0x%8X%8X", $randval32_1, $randval32_0;
    $numstring =~ s/ /0/;  # to not leave spaces in final composite number
    print OUT "$numstring\n";
}


#
# Compute hamming distance of 32 bit integers
#
sub hamming {
    my $val1 = $_[0];
    my $val2 = $_[1];

    my $or_val = ($val1 ^ $val2);
    my $bit_count = 0;

    while($or_val) {
	if($or_val&1) { $bit_count++; }
	$or_val = ($or_val >> 1);
    }

    return $bit_count;
}
