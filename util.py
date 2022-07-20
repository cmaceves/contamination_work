import os
import ast
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

def create_barplot(snv_output, metadata):
    """
    Function plots the silhoutte score 
    """
    print("create barplot")

    #merge
    merged_df = metadata.merge(snv_output, left_on="filename", right_on="index", how="right")
    
    labels = []
    counts = []
    for index, row in merged_df.iterrows():
        population_1 = row['percent_1']
        population_2 = row['percent_2']
        label_str = '%s/%s' %(population_1, population_2)
        sil = row['sil']
        
        #append to list
        labels.append(label_str)
        counts.append(sil)
        
    order_list = ['50.0/50.0', '40.0/60.0', '35.0/65.0', '30.0/70.0', '15.0/85.0', '10.0/90.0', '5.0/95.0']
    x = list(np.arange(0, len(labels)))
    sns.set_style("whitegrid")
    g = sns.scatterplot(x=x, y=counts, color='purple', s=100)
    plt.xticks(x,labels)
    plt.ylabel("Silhoutte Score")
    plt.savefig("./figures/barplot.png")

def create_regression_plot(snv_output, metadata):
    """
    Function takes in the cluster centers and the metadata dataframe
    and plot a regression of the two.
    """
    print('creating regression plot')

    #ground truth    
    x = []
    #experimental values
    y = []

    #merge
    merged_df = metadata.merge(snv_output, left_on="filename", right_on="index", how="right")
    for index, row in merged_df.iterrows():
        population_1 = row['reads_1'] / row['total_reads']
        population_2 = row['reads_2'] / row['total_reads']
        cluster_centers = ast.literal_eval(row["cluster_centers"])
        
        population_list = [population_1, population_2]
        population_list.sort(reverse=True)
        cluster_centers.sort(reverse=True)
        if len(cluster_centers) != len(population_list):
            print(row['index'], population_list, cluster_centers)
            if len(cluster_centers) > len(population_list):
                num_zeros = len(cluster_centers) - len(population_list)        
                population_list.extend([0.0]*num_zeros)        
           
        x.extend(population_list)
        y.extend(cluster_centers)

    plt.plot([0,1],[0,1], color='orange', linestyle='--')
    print(x,y)
    sns.regplot(x,y, color='purple')
    plt.xlabel("Population Frequencies")
    plt.ylabel("Cluster Center Values")
    plt.grid(visible=True, color='lightgrey')
    plt.savefig("./figures/regression_plot.png")     
    plt.close()
    
def create_lineplot(x, y, x_axis, y_axis, save_name):
    """
    Function creates a lineplot using seaborn.
    """

    pass    


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
        #TEST LINES 
        if fullcount % 100000 == 0:
            if fullcount > 0:
                print(fullcount)

        #get cigar values
        cigar = pileupread.cigartuples
        cigar = [list(x) for x in cigar]
        expand_cigar = []
        for temp_cig in cigar:
            expand_cigar.extend([temp_cig[0]]*temp_cig[1])

        #if the length is different their's an insertion
        ref_seq = list(pileupread.get_reference_sequence()) 

        total_query = list(pileupread.query_sequence)
        total_ref = pileupread.get_reference_positions(full_length=True)
        total_qualities = list(pileupread.query_qualities)
        alignment_object = pileupread.get_aligned_pairs(with_seq=True)
        for count, tr in enumerate(total_ref):
            if tr is None:
                ref_seq.insert(count, None)
        for count, cig in enumerate(expand_cigar):
            if cig == 2:
                total_query.insert(count, None)
                total_ref.insert(count, None)
                total_qualities.insert(count, None)
                ref_seq.insert(count, None)
                

        insertion=''
        deletion = ''
        on_insertion=False
        on_deletion=False
        
        #if these things aren't equal something is wrong
        if len(total_ref) != len(total_query) != len(total_qualities) != len(expand_cigar) != len(ref_seq):
            print("error.")
            sys.exit(1)

        for count, (r, q, qual, cigtype, al, rs) in enumerate(zip(total_ref, total_query, \
            total_qualities, expand_cigar, alignment_object, ref_seq)):           
            #deletions
            if cigtype == 2:
                on_deletion=True
                deletion+= al[2].upper()
                continue
            #soft clipped
            elif cigtype == 5 or cigtype == 4:
                continue
            #match
            elif cigtype == 0:
                if on_insertion:
                   r = r - len(insertion) 
                   nuc_add = '+'+insertion
                   insertion=''
                   ref = None
                   on_insertion=False
                elif on_deletion:
                    r = r - len(deletion)
                    nuc_add = '-'+deletion
                    deletion=''
                    ref = None
                    on_deletion=False
                else:
                    nuc_add = q.upper()
                    ref = rs.upper()
                
            #if we have an insertion
            elif cigtype == 1:
                on_insertion = True
                insertion += q.upper()
                continue
                              
            #if we are finished, we add the match either way 
            if  r not in position_dict:
                position_dict[r] = {}
                if ref is not None:
                    position_dict[r]['ref'] = {ref:1}
                else:
                    position_dict[r]['ref']={}
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
                if ref is not None and ref in position_dict[r]['ref']:
                    position_dict[r]['ref'][ref] += 1
                else:
                    if ref is not None:
                        position_dict[r]['ref'][ref] = 1


    return(position_dict)               
