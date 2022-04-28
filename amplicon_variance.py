import os
import sys
import ast
import json
import pysam
import argparse

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

from contaminant_analysis import calculate_positional_depths

def extract_amplicons(bam, primer_0, primer_1, primer_0_inner, primer_1_inner, noise_dict, pos_dict):
    """
    Parameters
    -----------
    bam : str
        Path to the bam file
    primer_0 : int
        The star pos of the primer
    primer_1 : int
        The end pos of the primer
    """
    print("look for amplicon between %s and %s" %(primer_0, primer_1))

    #encoded nucs, 0 means match ref
    encoded_nucs = {"A":1,"C":2,"G":3,"T":4,"N":5}

    #open the bam
    samfile = pysam.AlignmentFile(bam, "rb")
    
    #track how many amps we've found
    found_amps = 0
    
    #reads for this amp
    read_matrix = []
    #freq for this amp
    freq_matrix=[]
        
    #amp len
    amp_len = primer_1_inner - primer_0_inner
    amp_indexes = list(range(primer_0_inner, primer_1_inner))
    
    #make sure we don't get a pair twice
    seen_reads = []

    for pileupcolumn in samfile.pileup("NC_045512.2", start=primer_0, stop=primer_1):
        for pileupread in pileupcolumn.pileups:
            amp = [0]*amp_len
            freq = [0]*amp_len
            if pileupread.alignment.qname in seen_reads:
                continue
            if pileupread.alignment.reference_start >= primer_0 and pileupread.alignment.reference_end <= primer_1:
                positions = pileupread.alignment.get_reference_positions(full_length=True)
                query_seq = pileupread.alignment.query_sequence
                query_qual = pileupread.alignment.query_qualities
                seen_reads.append(pileupread.alignment.qname)
                cigar = pileupread.alignment.cigartuples
                cigar = [list(x) for x in cigar]
              
                store_nuc=''
                on_insertion=False
                for pcount,(pos,nuc,qual) in enumerate(zip(positions, query_seq, query_qual)):
                    #if we have an insertion we create a seperate encoding for it, and place it in
                    nuc = nuc.upper()
              
                    total=0 
                    for x in cigar:
                        if x[0] == 2 or x[0] == 5:
                            continue
                        total += x[1]
                        if total > pcount:
                            cigtype = x[0]
                            break
                    if cigtype == 1:
                        on_insertion = True
                        store_nuc += nuc.upper()
                        continue
                    if cigtype == 4:
                        continue
                    if pos != None and on_insertion is False:
                        pass
                    #we've hit the end of the insertion
                    elif pos != None and on_insertion is True:
                        new_value = max(encoded_nucs, key=encoded_nucs.get) 
                        nuc = '+'+store_nuc
                        encoded_nucs[nuc] = encoded_nucs[new_value]+1
                        on_insertion = False
                        pos = pcount - len(store_nuc)
                        store_nuc =''
                    if int(qual) < 20:
                        continue
                    if pos in amp_indexes:
                        if int(pos) in noise_dict:
                           noise = noise_dict[int(pos)]
                           if nuc in noise:
                                continue
                        loc = amp_indexes.index(pos)
                        amp[loc] = encoded_nucs[nuc]   
                        try:
                            #then we go find the overall freq
                            #print(pos_dict[str(pos)], nuc)
                            temp = pos_dict[str(pos)]['allele'][nuc]['count']/pos_dict[str(pos)]['total_depth']
                            if temp > 0.97:
                                continue
                            else:
                                freq[loc]=temp
                        except:
                            print("failed to file " + str(pos) + " in pos dict")
                 
                found_amps += 1           
            #print('fa: ', found_amps)
            if np.count_nonzero(amp) == 0:
                continue 
            #once we've added all the pos to the read array, we append to the matrix
            read_matrix.append(amp)
            freq_matrix.append(freq)
    #print("%s amplicons found" %found_amps)
    read_matrix=np.array(read_matrix)
    freq_matrix=np.array(freq_matrix)

    #estimated num groups 
    num_groups = []
    final_tuples = []
    poi = []
    #position of interest within this matrix
    relative_poi=[]

    #iterate by position
    for c, (sr, pos_array) in enumerate(zip(seen_reads,read_matrix.T)):
        filter_zeros = pos_array[pos_array>0]
        if len(filter_zeros) < 1:
            continue
        total_depth = len(pos_array)
        values, counts = np.unique(filter_zeros, return_counts=True)
        percent = [x/total_depth for x in counts]
        final = [z for z in zip(percent, values) if z[0] > 0.03]
        if len(final) > 0:
            relative_poi.append(c)
            poi.append(c+primer_0_inner)
        num_groups.append(len(final))
        final_tuples.append(final)
    
    """
    returns
    (1) average number of groups
    (2) average break down % wise if > 1
    (3) variance in groups across positions
    (4) positions that varied
    (5) number of positions with > 1 
    (6) reads in this amplicon that are useful
    """
    #for i,ng in enumerate(num_groups):
    #    if ng == 3:
    #        print(i+primer_0_inner)

    if len(num_groups) != 0:
        max_number_groups = max(num_groups)
        average_number_groups = np.average(num_groups)
        variance_number_groups = np.var(num_groups)
        percent_breakdown = [0.0]*round(average_number_groups)
        for thing in final_tuples:
            thing.sort(key=lambda y: y[0], reverse=True)
            for i,percent in enumerate(thing):
                if i >= round(average_number_groups):
                    break                
                percent_breakdown[i]+=percent[0]
        percent_breakdown = [x/len(num_groups) for x in percent_breakdown]
    else:
        max_number_groups = 0
        average_number_groups = 0
        percent_breakdown =[0]
        variance_number_groups = 0

    if read_matrix.ndim < 2:
        read_matrix = np.zeros((0,0)) 
     

    """
    print(
        Max Number of Groups: %s
        Average Number of Groups: %s
        Variance in Number of Groups: %s
        Number of Positions that Varied: %s
        Total Reads Looked at: %s
        Total Number of Pos Looked at: %s
        Percent Breakdown: %s 
     %(max_number_groups, average_number_groups, variance_number_groups, len(poi), read_matrix.shape[0], \
            read_matrix.shape[1], str(percent_breakdown)))
    """
    
    max_groups = []
    group_counts = []
    #if our max number of groups is > 1, let's do some clustering!
    if max_number_groups > 0:
        filter_matrix = freq_matrix.T[relative_poi]
        read_matrix_filt = read_matrix.T[relative_poi]
        
        groups = []
        groups2 = []
        for thing,thing2 in zip(filter_matrix.T, read_matrix_filt.T):
            if str(thing) not in groups:
                groups.append(str(thing))
                groups2.append(list(thing))
                group_counts.append(1)
            else:
                loc = groups.index(str(thing))
                group_counts[int(loc)] += 1
                continue 
        for a in groups2:
            a = [float(x) for x in a]
            max_groups.append(a)
    return(max_number_groups, average_number_groups, variance_number_groups, len(poi), poi, read_matrix.shape[0], read_matrix.shape[1], percent_breakdown, max_groups, group_counts)


def get_primers(primer_file):
    """
    Open a primer file and return a dict of lists with start/ends.
    
    Parameters
    ----------
    primer_file : str
        Path to the bed file.

    Returns
    -------
    primer_dict : dict
        Primer starts and ends in lists where keys are primer names.
    """
    
    content = []
    with open(primer_file, "r") as f:
        for line in f:
            content.append(line.strip().split())
    primer_names = []
    #iterate over the primer list
    for pl in content:
        #remove the forward, reverse aspect of the primer name
        pn = pl[3][:-1]
        primer_names.append(pn)
    
    unique_pn = list(np.unique(primer_names))
    primer_dict = {key:[0.0]*2 for key in unique_pn}
    primer_dict_inner = {key:[0.0]*2 for key in unique_pn}
   
    #iterate over the primer list again, this time save outer pos info
    for pl in content:
        pn = pl[3][:-1]
        direct = pl[3][-1]
        if direct == "F":
            val = pl[1]
            primer_dict[pn][0] = val
            primer_dict_inner[pn][0] = pl[2]
        if direct == "R":
            val = pl[2]
            primer_dict[pn][1] = val
            primer_dict_inner[pn][1] = pl[1]

    return(primer_dict, primer_dict_inner)


def parallel(all_bams):
    results = Parallel(n_jobs=30)(delayed(process)(bam) for bam in all_bams)

def main():
    all_files = [os.path.join("./spike_in/bam", x) for x in os.listdir("./spike_in/bam") if x.endswith('.bam')] 
    #all_files = [x for x in all_files if x != "./sp]
    parallel(all_files)
    #group_files_analysis()
    #process("./spike_in/bam/file_120.calmd.bam")
    """
    with open("saving_groups.json", "r") as jfile:
        data = json.load(jfile)

    amplicon_visualizations(data)
    """
   
def process(bam):
    
    basename = bam.split("/")[-1].split(".")[0]
    primer_file = "./spike_in/sarscov2_v2_primers.bed" 
    
    if os.path.isfile("./%s.json" %basename):
        return(0)

    if not os.path.isfile("./pos_depths/%s_pos_depths.json" %basename): 
        total_pos_depths = calculate_positional_depths(bam)
        with open("./pos_depths/%s_pos_depths.json" %basename, "w") as jfile:
            json.dump(total_pos_depths, jfile)
    else:
        with open("./pos_depths/%s_pos_depths.json" %basename, "r") as jfile:
            total_pos_depths = json.load(jfile)
     
    encoded_nucs = {"A":1,"C":2,"G":3,"T":4,"N":5}
    noise_dict = {}
    
    print_list = []
    #convert this into a positional noise deck
    for k,v in total_pos_depths.items():
        if int(k) in print_list:
            print(k, v)
        noise_dict[int(k)] = []
        total_depth = v['total_depth']
        alleles = v['allele']
        ref = max(v['ref'], key=v['ref'].get)
        noise_dict[int(k)].append(ref)
        for nt, count in alleles.items():
            if count['count']/total_depth < 0.03:
                noise_dict[int(k)].append(nt)
     
    #checkout this 
    #28800 and 28940
    #open and parse the bed file
    #gives us matches of primers by name
    primer_dict, primer_dict_inner  = get_primers(primer_file)
 
    p_bam=[]
    p_0=[]
    p_1=[]
    p_0_inner=[]
    p_1_inner=[] 
    count = 0
    file_level_dict = {}

    
    #this would be a good place to parallelize
    for k,v in primer_dict.items():   
        primer_0 = int(v[0])
        primer_1 = int(v[1])

        primer_0_inner = int(primer_dict_inner[k][0])
        primer_1_inner = int(primer_dict_inner[k][1])
        
        #if primer_0 != 21576:
        #    continue
        #if primer_0 != 14344:
        #    continue 
        #we didn't successfully find a primer pair
        if primer_0 == 0.0 or primer_1 == 0.0 or primer_0_inner == 0 or primer_1_inner ==0:
            continue    
        p_0.append(primer_0)
        p_1.append(primer_1)
        p_0_inner.append(primer_0_inner)
        p_1_inner.append(primer_1_inner)
        p_bam.append(bam)        
        max_num_groups, average_number_groups, variance_number_groups, len_poi, poi, num_reads_used, num_pos_used, \
            percent_breakdown, max_groups, group_counts = extract_amplicons(bam, primer_0, primer_1, primer_0_inner, primer_1_inner, noise_dict, \
        total_pos_depths)
        count += 1
        
        amplicon_level_dict = {"max_num_groups":max_num_groups, "average_number_groups":average_number_groups, \
            "variance_number_groups":variance_number_groups, "poi":poi, \
            "percent_breakdown": percent_breakdown, "num_reads_used": int(num_reads_used), \
            "num_pos_used": int(num_pos_used), "max_groups":max_groups, "group_counts":group_counts}
        file_level_dict[primer_0] = amplicon_level_dict
        #print(amplicon_level_dict, basename)    
    with open("%s.json" %basename, "w") as jfile:
        json.dump(file_level_dict, jfile)
    return(0)

def group_files_analysis():
    """
    Confusion matrix.
    """
    metadata_path = './spike_in/spike-in_bams_spikein_metadata.csv'
    metadata_df = pd.read_csv(metadata_path)
    meta_filenames = [x.split('.')[0] for x in metadata_df['filename'].tolist()]
    meta_abundance = [ast.literal_eval(x) for x in metadata_df['abundance(%)'].tolist()]
    meta_abundance_count = [len(x) for x in meta_abundance]
    meta_df = pd.DataFrame({"index":meta_filenames, "abundance_len":meta_abundance_count})

    all_files = [x for x in os.listdir("./spike_in/bam") if x.endswith('.bam')]
    json_filenames = [x.split('.')[0]+'.json' for x in all_files]
    seen = ["file_376.json"]   
    temp_dict = {}
    genes= [21563,25384]
    variant_read_covariance = []
    for file_output in json_filenames:
        if file_output not in seen:
            continue
        with open(file_output, 'r') as jfile:
            data = json.load(jfile)
            keys = list(data.keys())
            mng_all = []
            poi_all = []
            cluster_data=[]
            muts_all=[]
            group_muts=[] #max number of groups
            other_data=[]
            halotypes=[]
            mut_freq_dict = {}
            freq_groups = []
            for k,v in data.items():
                #related to the halotypes
                mng = v['max_groups']
                muts = v['poi']
                if len(mng) ==0:
                    continue
                for thing1, thing2 in zip(mng, muts):
                    if thing2 < 265 or thing2 > 29675:
                        mng.remove(thing1)
                        muts.remove(thing2)
                #man number of snvs
                group_muts.append(v['max_num_groups'])
               
                for m1 in np.array(mng).T:
                    halotypes.append(len(m1))
                muts_all.append(muts)
                mng_all.append(mng)

            #creat a dict that store the freq associated with each mut
            for mut, mng in zip(muts_all, mng_all):
                mng = np.array(mng).T.tolist()
                for c, (mut_temp, freq) in enumerate(zip(mut, mng)):
                    freq = [x for x in freq if x > 0]
                    mut_freq_dict[mut_temp] = freq

            freq_dict = {}
            for k,v in mut_freq_dict.items():        
                for freq in v:
                    if freq > 0.97 or freq < 0.03:
                        continue
                    found=False
                    #search against what we already have
                    for k2,v2 in freq_dict.items():
                        if abs(k2-freq) < 0.01:
                            found=True
                            if k not in freq_dict[k2]:
                                freq_dict[k2].append(k)
                            break
                    if found is False:
                        freq_dict[freq] = [k]
            pooled_muts = []
            for k,v in freq_dict.items():
                pooled_muts.extend(v)
            unique, counts = np.unique(pooled_muts, return_counts=True)
            
            flag_key=[]
            x =[]
            y = []
            mut_tag = []
            for k,v in freq_dict.items():
                #print(k,v)
                x.append(k)
                y.append(len(v))
                mut_tag.append(v)
                for f in v:
                    loc = list(unique).index(f)
                    if counts[loc] > 1:
                        flag_key.append(k)
            unique_freq = []
            for k,v in freq_dict.items():
                if k not in flag_key:
                    for f in v:
                        #if f > genes[0] and f < genes[1]:
                        unique_freq.append(k)
                        break
            unique_freq.sort()
            
            for x1,y1,m1, in zip(x,y,mut_tag):
                print(x1,y1,m1)
            sns.scatterplot(x,y)
            plt.savefig('trial.png')
            sys.exit(0)            
            
            max_snv = max(group_muts)
            max_halotype = max(halotypes)
            temp_dict[file_output.replace(".json","")]= {
                'max_halotypes':max_halotype, "max_snv":max_snv}
             
    df_outcome = pd.DataFrame(temp_dict).T
    df_outcome = df_outcome.reset_index()
    final_df = df_outcome.merge(meta_df, on='index', how='left')
    final_df.to_csv("snv_output.csv")
    print(final_df) 
 
    sns.set_style("whitegrid")
    #sns.scatterplot(x='abundance_len',y=0, data=final_df)
    
    sns.swarmplot(x='abundance_len',y='max_snv', data=final_df)
    plt.ylim(0, 10)
    plt.xlabel("Number of Populations")
    plt.ylabel("Maximum Number of 'Frequency profiles'")
    plt.title("Max Number of frequency profiles versus GT Populations") # You can comment this line out if you don't need title
    plt.savefig("snv_max_real.png")

def amplicon_visualizations(data):
    """
    File level visualization code.
    Bar graph of groups across the amplicons, with variance as error bars.
    Stacked bar graph of group percent breakdowns.
    """
    primer_0 = []
    groups = []
    variance = []
     
    for k,v in data.items():
        primer_0.append(k)
        groups.append(v['average_number_groups'])
        variance.append(v['variance_number_groups'])
           
    sns.barplot(x=primer_0, y=groups)
    plt.xticks([])
    plt.savefig("barplot.png")

    
    

 
if __name__ == "__main__":
    main()
