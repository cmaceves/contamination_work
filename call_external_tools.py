"""
Python wrappers for ivar and nextstrain.
"""
import os
silent=False

def call_consensus(filename, output_filename, threshold):
    """
    Given an input file, an ouput path, and a threshold, call consensus on a file.
    """
    cmd = "samtools mpileup -A -d 0 -Q 0 %s | ivar consensus -p %s -t %s" %(filename, output_filename, threshold)
    os.system(cmd)

def call_variants(bam, basename, reference_filepath, variants_output_dir):
    """
    Parameters
    ----------
    bam : str
        Full path to bam file.
    basename : str
        The name of the sample.
    reference_filepath : str
        Path to the reference .fasta file.
    variants_output_dir : str
        The directory in which to output the variants .tsv

    Function takes in a bam file and uses command line to call ivar variants.
    """

    print("calling ivar variants on", basename)

    cmd = "samtools mpileup -aa -A -d 0 -B -Q 0 %s | ivar variants -p %s/variants_%s -q 20 -r %s" \
            %(bam, variants_output_dir, basename, reference_filepath)
    os.system(cmd)
def call_getmasked(bam, basename, variants_output_dir, bed_filepath, primer_pair_filepath, \
        output_dir):
    """
    Parameters
    ----------
    bam : str
        Full path to bam file.
    basename : str
        The name of the sample.
    variants_output_dir : str
        The directory in which to output the variants .tsv
    bed_filepath : str
        The full path to the bed file.
    primer_pair_filepath : str
        The full path to the primer pair files.
    output_dir : str
        The directory to output the .txt file from ivar getmasked.
    """
    variants_filename = os.path.join(variants_output_dir, "variants_"+basename+".tsv")
    if os.path.isfile(variants_filename):
        pass
    else:
        return(1)

    if not silent:
        print("calling ivar getmasked on", basename)

    cmd = "ivar getmasked -i %s -b %s -f %s -p %s/masked_%s" %(variants_filename, bed_filepath, \
            primer_pair_filepath, output_dir, basename)
    os.system(cmd)

def call_pangolin(multifasta, output_filename):
    """
    Given a multi-fasta and output filename, align and output lineages.
    """
    cmd = "pangolin %s --outfile %s --alignment" %(multifasta, output_filename)
    os.system(cmd)


