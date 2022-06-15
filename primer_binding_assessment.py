import os
import sys
import copy
import json
import pysam

import numpy as np
from joblib import Parallel, delayed

#import functions from other files
from amplicon_variance import get_primers

def find_primer_muts(bam, primer_0, primer_1, primer_0_inner, primer_1_inner, total_pos_depths, \
        masked_primers, name):
    #open the bam
    samfile = pysam.AlignmentFile(bam, "rb")
    
    #make sure we don't get a pair twice
    seen_reads = []
    names_taken = []
    found_amps = 0

    dropped_reads = []
    
    for pileupcolumn in samfile.pileup("NC_045512.2", start=primer_0, stop=primer_1):
        for pileupread in pileupcolumn.pileups:
            reverse = pileupread.alignment.is_reverse
            if reverse:
                primer_name = 'R'
            else:
                primer_name = 'F'
            if pileupread.alignment.qname+"_"+primer_name in seen_reads:
                continue
            #sys.exit(0)
            if pileupread.alignment.reference_start >= primer_0 and pileupread.alignment.reference_end <= primer_1:
                seen_reads.append(pileupread.alignment.qname+"_"+primer_name)
                found_amps += 1
                
                reverse = pileupread.alignment.is_reverse
                mismatch=False
                if reverse:
                    primer_name = name+'R'
                else:
                    primer_name = name+'F'
                #test line, the forward primer for 23,514 is messed up
                if not reverse:
                    mismatch=True

                #targeted_pos = list(range(primer_1_inner, primer_1))
                targeted_pos = list(range(primer_0, primer_0_inner))
                #print(targeted_pos)
                positions = pileupread.alignment.get_reference_positions(full_length=True)

                query_seq = pileupread.alignment.query_sequence
                query_qual = pileupread.alignment.query_qualities
                seen_reads.append(pileupread.alignment.qname)
                cigar = pileupread.alignment.cigartuples
                cigar = [list(x) for x in cigar]
                new_positions = [0]*len(query_seq)
                
                #make sure we replace the None with the positons that aren't insertions
                for c,p in enumerate(positions):
                    cig_pos=0
                    last_pos=0
                    for cig in cigar:
                        cig_pos += cig[1]
                        if last_pos <= c <= cig_pos:
                            if cig[0] != 1:
                                new_positions[c]=-1
                            else:
                                new_positions[c]=0
                        last_pos += cig[1]
                for c, p in enumerate(positions):
                    #it's not an insertion
                    if new_positions[c] != 0:
                        if p is not None:
                            new_positions[c]=p
                        else:
                            #iterate forward to find closest number
                            try:
                                for s,t in enumerate(positions[c:]):
                                    if t is not None:
                                        new_positions[c] = t-s 
                                        break
                            #iterate backward to find cloest number
                            except:
                                pass
                            try:
                                new_list = copy.deepcopy(positions[:c+1])
                                new_list.reverse()
                                for s,t in enumerate(new_list):
                                    if t is not None:
                                        new_positions[c] = t+s
                                        break
                            except:
                                pass
                   
                seen_primer_nucs = 0
                mutated_primer_nucs = 0
                #print(targeted_pos, new_positions)
                #sys.exit(0)
                print("\n")
                for pcount,(pos,nuc,qual) in enumerate(zip(new_positions, query_seq, query_qual)):                     
                    if pos in targeted_pos:
                        seen_primer_nucs += 1
                        try:
                            ref = max(total_pos_depths[str(pos)]['ref'], key=total_pos_depths[str(pos)]['ref'].get)
                        except:
                            continue
                        #check if the primer nuc matches the ref
                        if ref.upper() != nuc.upper() and mismatch:
                            mutated_primer_nucs += 1
                            print(pileupread.alignment.qname, pos, ref.upper(), nuc.upper())
                        
                #if seen_primer_nucs == 0:
                #    continue
                #percent_mut_nucs = mutated_primer_nucs/seen_primer_nucs
                #if percent_mut_nucs > 0:
                if reverse:
                    direction='R'
                else:
                    direction='F'
                if mismatch:
                    direction = 'F'
                    #print(pileupread.alignment.qname, pileupread.alignment.is_reverse)
                    dropped_reads.append(pileupread.alignment.qname + " %s"%direction)
                #print(cigar, primer_0, percent_mut_nucs, seen_primer_nucs, mutated_primer_nucs, ref.upper(), nuc.upper())
                
    return(dropped_reads)

def main():
    """
    Looks at each of the primer binding sites and check for mutations.
    """

    file_test_list = ["file_124","file_125","file_127"]
    file_test_bam = [os.path.join("../spike_in",x+"_sorted.calmd.bam") for x in file_test_list]
    file_test_list = ['file_124']
    file_test_bam = ["../untrimmed_bam/file_124.new.sorted.bam"]
    process(file_test_bam[0], file_test_list[0])
    #Parallel(n_jobs=3)(delayed(process)(bam,basename) for bam,basename in zip(file_test_bam, file_test_list))

def process(bam, basename):    
    print(basename)
    primer_file = "../sarscov2_v2_primers.bed"
    primer_dict, primer_dict_inner  = get_primers(primer_file)
    
    #get the primers that are masked
    masked_primers=["covid19genome_200-29703_s20720_D_32F"]

    #with open("../untrimmed_bam/masked_124.txt","r") as mfile:
    #    for line in mfile:
    #        masked_primers.append(line.strip())


    with open("../pos_depths/%s_pos_depths.json" %basename, "r") as jfile:
        total_pos_depths = json.load(jfile)
    
    temp_dict = {}
    #iterate through each of the primer sets
    for k,v in primer_dict.items():
        
        primer_0 = int(v[0])
        primer_1 = int(v[1])

        primer_0_inner = int(primer_dict_inner[k][0])
        primer_1_inner = int(primer_dict_inner[k][1])

        if primer_0 == 0.0 or primer_1 == 0.0 or primer_0_inner == 0 or primer_1_inner ==0:
            continue
        if primer_0 != 23514:
            continue
        dropped_reads = find_primer_muts(bam, primer_0, primer_1, primer_0_inner, primer_1_inner, \
                total_pos_depths, masked_primers, k)
        temp_dict[primer_0] = dropped_reads
    
    with open("../primer_drops/primer_%s.json" %basename, "w") as jfile:
        json.dump(temp_dict, jfile)
    return(0)
if __name__ == "__main__":
    main()
