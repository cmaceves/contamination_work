"""
Python wrappers for ivar and nextstrain.
"""
import os
silent=False


def retrim_bam_files(bam, basename, output_dir, ref_seq, primer_bed):
    """
    Retrim the bam files.

    Parameters
    ----------
    bam : str
        Full path to the bam file.
    basename : str
        The name of the file to conserve in processing.
    output_dir : str
        The full path the to the output directory.
    ref_seq : str
        The full path to the reference sequence. 
    primer_bed : str
        The full path to the primer bed file.

    Returns
    -------
    final_bam : str
        The full path to the final bam file after retrimming.
    """
   
    intermediate_name = os.path.join(output_dir, basename + ".namesorted.bam")
    fastq1 = os.path.join(output_dir, basename + "_r1.fq")
    fastq2 = os.path.join(output_dir, basename + "_r2.fq")
    intermediate_sorted = os.path.join(output_dir, basename + ".sorted.intermediate.bam")
    intermediate_trimmed = os.path.join(output_dir, basename + ".trimmed.intermediate")
    intermediate_tag = os.path.join(output_dir, basename + ".pretag.bam")
    final_bam = os.path.join(output_dir, basename + ".final.bam")

    cmd = "samtools sort -n %s -o %s" %(bam, intermediate_name)
    os.system(cmd)
    cmd1 = "bedtools bamtofastq -i %s -fq %s -fq2 %s" %(intermediate_name, fastq1, fastq2)
    os.system(cmd1) 
    cmd2 = "bwa mem -t 32 %s %s %s| samtools view -b -F 4 -F 2048 | samtools sort -o %s" \
        %(ref_seq, fastq1, fastq2, intermediate_sorted)
    os.system(cmd2)
    cmd3 = "ivar trim -b %s -p %s -i %s" %(primer_bed, intermediate_trimmed, intermediate_sorted)
    os.system(cmd3)
    cmd4 = "samtools sort -o %s %s" %(intermediate_tag, intermediate_trimmed+".bam")     
    os.system(cmd4)
    cmd5 = "samtools index %s" %(intermediate_tag)
    os.system(cmd5)
    cmd6 = "samtools calmd -b %s %s > %s" %(intermediate_tag, ref_seq, final_bam)
    os.system(cmd6)
    cmd7 = "samtools index %s" %(final_bam)
    os.system(cmd7)    

    #print(cmd, "\n", cmd1, "\n", cmd2, "\n", cmd3, "\n", cmd4, "\n", cmd5, "\n", cmd6, "\n", cmd7)
    
    os.system("rm %s" %intermediate_name)
    os.system("rm %s" %fastq1)
    os.system("rm %s" %fastq2)
    os.system("rm %s" %intermediate_sorted)
    os.system("rm %s" %(intermediate_sorted+".bai"))
    os.system("rm %s" %intermediate_trimmed+".bam")
    os.system("rm %s" %(intermediate_tag+".bai"))
    os.system("rm %s" %intermediate_tag)
    
    return(final_bam)

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


