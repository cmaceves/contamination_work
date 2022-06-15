import os
import math
import copy
import time
import sys
import statistics
import pysam
import pickle
import json
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def calculate_positional_depths(bam, dropped_reads=None):
    """
    Creates a dict for each position with nucs, depths, and qual.
    
    Parameters
    ----------
    bam : str
        Path to the bam file to iterate.
    dropped_reads : str
        List of reads to exclude from the depth count.

    Returns
    -------
    position_dict : dict
        Nucs with depths and qual at each position.
    """
    print("Calculating positional depths: ", bam)
    query_names = []
    samfile = pysam.AlignmentFile(bam, "rb")
      
    unique_headers = []
    position_dict = {}    
    seen_reads = [] 
    
    past_r = ''
    
    for fullcount, pileupread in enumerate(samfile):
        #if we have dropped reads, test if this read is in them
        if dropped_reads is not None:
            if pileupread.is_reverse:
                letter='R'
            else:
                letter='F'
            if pileupread.qname+" "+letter in dropped_reads:
                continue
                
        if fullcount % 100000 == 0:
            if fullcount > 0:
                print(fullcount)
    
        #get cigar values
        cigar = pileupread.cigartuples
        cigar = [list(x) for x in cigar]
       
        #if the length is different their's an insertion
        ref_seq = pileupread.get_reference_sequence()
        ref_pos = pileupread.get_reference_positions()

        seq = pileupread.query_alignment_sequence 
        total_ref = pileupread.get_reference_positions(full_length=True)
        
        insertion=''
       
        finished=False
        on_insertion=False       
        for count, (r, q, qual) in enumerate(zip(total_ref, pileupread.query_sequence, \
            pileupread.query_qualities)):
            total=0                                   

            #this tells you what it should be based on the cigar, but if it doesn't land
            #in the del range you remove prior deletiosn from the cig
            for x in cigar:
                if x[0] == 2 or x[0] == 5:
                    continue
                total += x[1]
                if total > count:
                    cigtype = x[0]
                    break
            if cigtype == 0:
                if on_insertion:
                   stored_nuc = insertion
                   past_r = total_ref[count-len(stored_nuc)-1]                      
                    
                nuc_add = q.upper() 
                rloc = ref_pos.index(r)
                ref = ref_seq[rloc].upper()
                finished=True
                if past_r is None:
                    past_r = r-1

            #if we have an insertion we have no reference
            elif cigtype == 1:
                on_insertion = True
                finished=False
                insertion += q.upper()                
            #if we have a deletion, the reference is the same as the nucs
            elif cigtype == 2:
                continue
            elif cigtype == 4:
                continue
            else:
                print("cig type not accounted for ", cigtype)
                sys.exit(0)

            #if we aren't finished, keep iterating
            if not finished:
                continue
            if r is None or str(r) == "null":
                print(r, count, total_ref, cigar)
            #if we are finished, we add the match either way
            if not on_insertion:
                if  r not in position_dict:
                    position_dict[r] = {}
                    position_dict[r]['ref'] = {ref:1}
                    position_dict[r]['allele'] = {nuc_add:{"count": 1, 'qual': qual}}
                    position_dict[r]['total_depth'] = 1
                else: 
                    #check if we've seen this nuc before
                    if nuc_add in position_dict[r]["allele"]:
                        position_dict[r]['allele'][nuc_add]["count"] += 1
                        position_dict[r]['total_depth'] += 1 
                        position_dict[r]['allele'][nuc_add]["qual"] += qual    
                    #else we will add the letter            
                    else:
                        position_dict[r]['allele'][nuc_add] = {"count":1, 'qual':qual}
                        position_dict[r]['total_depth'] += 1
                    #handle the ref addition
                    if ref in position_dict[r]['ref']:
                        position_dict[r]['ref'][ref] += 1
                    else:
                        position_dict[r]['ref'][ref] = 1
            if on_insertion:
                if past_r is None:
                    print(stored_nuc, r, total_ref, cigar)
                r = past_r              
                #insertion at an unseen position  
                if r not in position_dict and on_insertion:
                    store_nuc = "+"+stored_nuc
                    position_dict[r] = {}
                    position_dict[r]['ref'] = {}
                    position_dict[r]['allele'] = {store_nuc:{"count": 1, 'qual': qual}}
                    position_dict[r]['total_depth'] = 1
                    on_insertion=False
                    insertion=''
                    stored_nuc=''
                #we have an insertion at a position we've seen before
                elif r in position_dict and on_insertion:
                    store_nuc = "+"+stored_nuc
                    if store_nuc in position_dict[r]["allele"]:
                        position_dict[r]['allele'][store_nuc]["count"] += 1
                        position_dict[r]['total_depth'] += 1 
                        position_dict[r]['allele'][store_nuc]["qual"] += qual    
                    #else we will add the letter            
                    else:
                        position_dict[r]['allele'][store_nuc] = {"count":1, 'qual':qual}
                        position_dict[r]['total_depth'] += 1
                    on_insertion=False
                    insertion=''
                    stored_nuc=''
 
    return(position_dict)               
