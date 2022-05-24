import os
import sys
import ast
import copy
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
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from contaminant_analysis import calculate_positional_depths

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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
    
    #qualities for this amp
    quality_matrix = []
 
    #amp len
    amp_len = primer_1_inner - primer_0_inner
    amp_indexes = list(range(primer_0_inner, primer_1_inner))
       
    #make sure we don't get a pair twice
    seen_reads = []
    names_taken = [] 
    for pileupcolumn in samfile.pileup("NC_045512.2", start=primer_0, stop=primer_1):
        for pileupread in pileupcolumn.pileups:
            amp = [-1]*amp_len
            freq = [-1]*amp_len
            quality = [0.0] *amp_len
            if pileupread.alignment.qname in seen_reads:
                
                continue
            if pileupread.alignment.reference_start >= primer_0 and pileupread.alignment.reference_end <= primer_1:
                seen_reads.append(pileupread.alignment.qname)
                
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
                    #test code
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
                    #don't count it due to low base quality
                    """
                    if int(qual) < 20:
                        if pos in amp_indexes:
                            loc = amp_indexes.index(pos)
                            amp[loc] = 0.0
                            freq[loc]=0.0
                        continue
                    """   
                    if pos in amp_indexes:
                        if int(pos) in noise_dict:
                            noise = noise_dict[int(pos)]
                            #this base is noise or reference
                            if nuc in noise:
                                loc = amp_indexes.index(pos)
                                amp[loc] = 0.0
                                freq[loc]=0.0
                                continue
                        loc = amp_indexes.index(pos)
                        amp[loc] = encoded_nucs[nuc]
                        quality[loc] += qual   
                        try:
                            #then we go find the overall freq
                            #print(pos_dict[str(pos)], nuc)
                            temp = pos_dict[str(pos)]['allele'][nuc]['count']/pos_dict[str(pos)]['total_depth']
                            if temp > 0.97:
                                loc = amp_indexes.index(pos)
                                amp[loc] = 0.0
                                freq[loc] =0.0
                                quality[loc]=0.0
                                continue
                            else:
                                freq[loc]=temp
                        except:
                            print("failed to find " + str(pos) + " in " + bam + " pos dict")
            
                found_amps += 1           
                #print('fa: ', found_amps)
                if np.count_nonzero(amp) == 0:
                    continue 
                names_taken.append(pileupread.alignment.qname)
                #once we've added all the pos to the read array, we append to the matrix
                read_matrix.append(amp)
                freq_matrix.append(freq)
                quality_matrix.append(quality)
    print('found amps: ', found_amps)
    
    #return(found_amps)

    #print("%s amplicons found" %found_amps)
    read_matrix=np.array(read_matrix)
    
    freq_matrix=np.array(freq_matrix)
    quality_matrix=np.array(quality_matrix)

    #estimated num groups 
    num_groups = []
    final_tuples = []
    poi = []
    #position of interest within this matrix
    relative_poi=[]
   
    #iterate by position
    for c, (pos_array, qual_array) in enumerate(zip(read_matrix.T, quality_matrix.T)):
        filter_zeros = pos_array[pos_array>0]
        if len(filter_zeros) < 1:
            continue
        #remover blank values from quality
        filter_qual = qual_array[qual_array>0]
        #find the average quality here
        avg_qual = np.average(filter_qual)
        if avg_qual < 20:
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
    #print(poi) 
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
        quality_matrix_filt = quality_matrix.T[relative_poi]
        groups = []
        groups2 = []

        for thing,thing2,qual in zip(filter_matrix.T, read_matrix_filt.T, quality_matrix.T):
            if -1 in thing:
                continue
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
            
    group_percents = [x/found_amps for x in group_counts]
        
    for i,(mg, gc) in enumerate(zip(max_groups, group_percents)):
        if np.count_nonzero(mg) == 0:
            max_groups.remove(mg)
            group_percents.remove(gc)
    print(poi, group_percents, max_groups, found_amps)
    return(poi, group_percents, max_groups,found_amps)


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
    results = Parallel(n_jobs=35)(delayed(process)(bam) for bam in all_bams)
    

def parse_snv_output(csv_filename):
    """
    Gets filenames and thresholds from the .csv file output.
    Returns dict with filenames and thresholds.
    """
    file_threshold_dict = {}
    df = pd.read_csv(csv_filename)
    for index, row in df.iterrows():
        thresh = row['threshold']
        filename = row['index']
        other_thresh = row['threshold_amb']
        bam_filename = "./spike_in/all_bams/" + filename + "_sorted.calmd.bam"
        file_threshold_dict[bam_filename] = {"threshold": round(thresh,2), "output": "./consensus/"+\
            filename +"_"+ str(round(thresh,2)), 'other_thresh':round(other_thresh,2), "other_output": \
            "./consensus/"+filename+"_"+str(round(other_thresh,2))}

    return(file_threshold_dict)

def main():
    all_files = [os.path.join("./spike_in/all_bams", x) for x in os.listdir("./spike_in/all_bams") if x.endswith('_sorted.calmd.bam')] 
    #all_files = ["./spike_in/all_bams/file_300_sorted.calmd.bam", "./spike_in/all_bams/file_301_sorted.calmd.bam"]
    #parallel(all_files)
 
    #this line is used for testing
    #process("./spike_in/all_bams/file_300_sorted.calmd.bam")

    #creates the .csv file with thresholds and other info 
    #group_files_analysis()

    #sys.exit(0)
    #parses the tsv containing thresholding information
    file_threshold_dict = parse_snv_output("snv_output.csv")
    
    """
    #calls consensus using the above parsed thresholds
    for k, v in file_threshold_dict.items():
        print(k, " opt thresh")
        threshold = v['threshold']
        output_filename = v['output']
        other_thresh = v['other_thresh']
        other_output=v['other_output']
         
        if os.path.isfile(output_filename + ".fa"):
            continue
        if float(threshold) > 0.50:
            call_consensus(k, output_filename, threshold)            
        if os.path.isfile(other_output + ".fa"):
            continue
        if float(other_thresh) > 0.5:
            call_consensus(k, other_output, other_thresh)            
    
    #sys.exit(0) 
    #calls consensus using an array of thresholds 0.5-0.95 every 0.05 
    possible_consensus = [round(x,2) for x in list(np.arange(0.5,1.00,0.05))]
    for k,v in file_threshold_dict.items():
        for pc in possible_consensus:
            print(k, pc)
            original_output = v['output']
            new_output = '_'.join(original_output.split('_')[:-1]) + "_" + str(pc)
            if os.path.isfile(new_output + ".fa"):
                continue
            call_consensus(k, new_output, pc)
    """
    metadata_path = './spike_in/spike-in_bams_spikein_metadata.csv'
    metadata_df = pd.read_csv("snv_output.csv")
    analyze_nextstrain("nextclade.tsv", metadata_df)

def analyze_nextstrain(filename, metadata):
    """
    Parse apart nextclade file and analyze the consensus calling data.
    """
    s_gene_dict = {"alpha":[69,70,144,501,570,614,681,716,982,1118], \
        "beta":[80,215,241,243,417,484,501,614,701], \
        "delta":[19,156,157,158,452,478,614,681,950], \
        "gamma":[18,20,26,138,190,417,484,501,614,655,1027,1176]}
    
    df = pd.read_table(filename)
    test_file = "file_110"
    
    global_tpr = []
    global_fpr = []
    global_percents = []
    global_strains = []
    global_thresh = []
    gamb = []
    gtamb = []
    for index, row in df.iterrows():
        
        aa_sub = row['aaSubstitutions']
        if str(aa_sub) != 'nan':
            aa_sub = aa_sub.split(',')
        else:
            aa_sub = []
        
        aa_del = row['aaDeletions']
        if str(aa_del) != 'nan':
            aa_del = aa_del.split(',')
        else:
            aa_del =[]
        aa_ins = row['aaInsertions']
        if str(aa_ins) != 'nan':
            aa_ins = aa_ins.split(',')
        else:
            aa_ins=[]
        seqName = row['seqName']      
        clade = row['clade']
        nextclade_pango = row['Nextclade_pango'] 
 
        fileName = '_'.join(seqName.split("_")[1:3])
        #if fileName != test_file:
        #    continue
        
        act_thresh = float(seqName.split("_")[3])
        
        meta_row = metadata.loc[metadata['index']==fileName]
        try:
            variants = ast.literal_eval(meta_row['variant'].tolist()[0])
        except:
            continue
        variants = [x.lower() for x in variants]
        relative_percent = ast.literal_eval(meta_row['abundance'].tolist()[0])
       
        if 80 not in relative_percent:
            continue

        if variants[1] == 'aaron':
            continue
        expected_muts = [] 
        for v in variants:
            if v == 'aaron':
                
                expected_muts.append([])
            else:
                expected_muts.append(s_gene_dict[v])
        #these things only need to be added once per file
        if act_thresh == 0.5:
            tamb = list(meta_row['threshold'])[0]
            gtamb.append(tamb)
            amb = list(meta_row['threshold_amb'])[0]
            gamb.append(amb)
        
        muts_to_see = []
        actual_strain = ''
        #print(variants)
        #give the relative percents, which mutations would we expect to see?
        for em,rp,v in zip(expected_muts,relative_percent,variants):
            if rp > 50.0:
                muts_to_see.extend(em)
                actual_strain = v
        
        #print(clade, nextclade_pango, actual_strain)
        #print(nextclade_pango)
        
        found_muts = []
        tp = 0
        fp = 0
        for sub in aa_sub:
            gene = sub.split(":")[0]
            if gene != "S":
                continue
            mut = sub.split(":")[1]           
            found_muts.append(int(mut[1:-1]))    
        for dele in aa_del:
            gene = dele.split(":")[0]
            if gene != "S":
                continue
            mut = dele.split(":")[1]          
            found_muts.append(int(mut[1:-1]))    
            
        for ins in aa_ins:
            gene = ins.split(":")[0]
            if gene != "S":
                continue
            mut = ins.split(":")[1]           
            found_muts.append(int(mut[1:-1]))    
       
        found_muts.sort()  
       
        #print(act_thresh)
        #print(actual_strain)
        #print(found_muts)
        #print(muts_to_see)
        
        #cases in which we should see an upper group
        if actual_strain != '' and actual_strain != 'aaron':
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            #for the upper most group, how many 
            for fm in found_muts:
                if fm in muts_to_see:
                    tp += 1                    
                else:
                    fp += 1
            for ms in muts_to_see:
                if ms not in found_muts:
                    fn += 1
                       
            #if tp == 0 or fn == 0:
            #    continue
            print(muts_to_see, found_muts, variants, act_thresh) 
            global_tpr.append(tp/(tp+fn))
            global_fpr.append(tp/(tp+fn))
            global_percents.append(str(variants))
            global_thresh.append(act_thresh)
    #print(global_thresh)
   
    sns.jointplot(x=global_thresh, y=global_tpr, kind='hex', color='black', gridsize=10)
   
    plt.axvline(x=float(np.average(gtamb)), color='red', linewidth=2)
    plt.axvline(x=float(np.average(gamb)), color='blue', linewidth=2)
    plt.axvline(x=0.80, color='orange',linewidth=2 )
    #sns.scatterplot(x=global_thresh, y=global_tpr, hue=global_percents)
    plt.xlabel("thresholds")
    plt.xlim(0.5,1)
    plt.ylabel("true positive rate")
    plt.title("20/80 files")
    plt.tight_layout() 
    plt.savefig("test_scatter.png")
    plt.close()
    
    """
    #get aa level changes
    #print(df.columns)
    aa_sub = df['aaSubstitutions'].tolist()
    aa_del = df['aaDeletions'].tolist()
    aa_ins = df['aaInsertions'].tolist()
    """


def process(bam):    
    basename = bam.split("/")[-1].split(".")[0].replace("_sorted","")
    
    primer_file = "./spike_in/sarscov2_v2_primers.bed"  
    
    #check if we've already processed this file
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
    
              
        #if primer_0 != 24388:
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
 
        poi, groups_percents, max_groups,found_amps = extract_amplicons(bam, primer_0, primer_1, primer_0_inner, primer_1_inner, noise_dict, total_pos_depths)

        amplicon_level_dict = { "poi":poi, \
            "groups_percents": groups_percents, "found_amps": int(found_amps), \
            "max_groups":max_groups}
        file_level_dict[primer_0] = amplicon_level_dict
         
    with open("%s.json" %basename, "w") as jfile:
        json.dump(file_level_dict, jfile)
    return(0)

def group_files_analysis():  
    #load metadata
    metadata_path = './spike_in/spike-in_bams_spikein_metadata.csv'
    metadata_df = pd.read_csv(metadata_path)
   
    #get columns for downstream.... could be written better
    meta_filenames = [x.split('.')[0] for x in metadata_df['filename'].tolist()]
    meta_abundance = [ast.literal_eval(x) for x in metadata_df['abundance(%)'].tolist()]
    meta_abundance_count = [len(x) for x in meta_abundance]
    meta_df = pd.DataFrame({"index":meta_filenames, "abundance_len":meta_abundance_count, \
        'abundance':[str(x) for x in meta_abundance], "variant":metadata_df['variant'].tolist()})

    all_files = [x for x in os.listdir("./spike_in/all_bams") if x.endswith('sorted.calmd.bam')]
    json_filenames = [(x.split('.')[0]+'.json').replace("_sorted","") for x in all_files]
    
    seen = ["file_300.json", "file_301.json"]   
    seen=[]
    temp_dict = {}
   
    variant_read_covariance = []
    all_mng_data = []

    #iterate through every output result
    for file_output in json_filenames:
        if not os.path.isfile(file_output):
            continue
        if file_output in seen:
            continue
        basename = file_output.split(".")[0]

        #load total positional depths
        with open(file_output, 'r') as jfile:
            data = json.load(jfile)
            keys = list(data.keys())
            mng_all = []
            poi_all = []
            cluster_data=[]
            store_max_freq = []
            muts_all=[]
            group_muts=[] #max number of groups
            other_data=[]
            halotypes=[]
            mut_freq_dict = {}
            freq_groups = []
            collapsed_read_max = 0
            for k,v in data.items():
                
                total_amp_depth = v['found_amps']
                 
                #related to the halotypes
                if total_amp_depth < 10:
                    continue
                percents = v['groups_percents']
                mng = v['max_groups']
                muts = v['poi']
                #print(file_output, mng, muts, percents)
                #this amplicon contained no useful info about variance
                if len(mng) ==0:
                    continue
                
                #not in coding region
                for thing1, thing2 in zip(mng, muts):
                    if thing2 < 265 or thing2 > 29675:
                        mng.remove(thing1)
                        muts.remove(thing2)
       
                keep_halotype_index = [i for i,c in enumerate(percents) if c > 0.03]
                mng = [x for i,x in enumerate(mng) if i in \
                    keep_halotype_index and np.count_nonzero(x) != 0]
                
                if len(mng) == 0:
                    continue
                for m1 in np.array(mng).T:
                    halotypes.append(len(m1))
                    num_halo = len(m1)
                    store_max_freq.append(mng)
                   
                #print(mng)
                #here we collapse in this way
                # (1) frequency within reads
                # (2) frequency between reads
                #store reads by freq profile
                stored_read_profiles = []
                for ab in mng:
                    collapse=[]
                    nonzero = [x for x in ab if x > 0]
                    if len(nonzero) == 0:
                        continue
                    for nz in nonzero:
                        found=False
                        for c in collapse:
                            if abs(c-nz) < 0.01:
                                found=True
                                break
                        if not found:
                            collapse.append(nz)
                    stored_read_profiles.extend(collapse)
                #print(stored_read_profiles)
                collapsed_read_profiles=[]
                for item in stored_read_profiles:
                    found=False
                    for item2 in collapsed_read_profiles:
                        if abs(item-item2) < 0.01:
                            found=True
                            break
                    if not found:
                        collapsed_read_profiles.append(item)
               
                if len(collapsed_read_profiles) > collapsed_read_max:
                    collapsed_read_max = len(collapsed_read_profiles) 
                    prof_of_choice = collapsed_read_profiles 
                
                 
                #test code for me only
                #used to look at potential halotype clustering
               
                #if file_output == "file_16.json":
                """
                if num_halo == 4:
                    print('in it', file_output)
                    #print(num_halo)
                    print('percents ', percents)
                    for ab in mng:
                        print([round(x,2) for x in ab])
                    #print(np.array(mng).shape, file_output, k)
                    print('muts ', muts)
                    #sys.exit(0)      
                """
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

            #pooling all mutation values
            for k,v in freq_dict.items():
                pooled_muts.extend(v)
            unique, counts = np.unique(pooled_muts, return_counts=True)
            
            flag_key=[]
            x =[]
            y = []
            mut_tag = []
            for k,v in freq_dict.items():
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
            zipped = zip(x,y,mut_tag)
            zipped = sorted(zipped, key = lambda x:x[1])
            all_mng_data.append(zipped)
            x,y,mut_tag = zip(*zipped)
            c_data = []
            all_muts_count = sum(y)
            for x1,y1,m1, in zip(x,y,mut_tag):
                c_data.extend([x1])
                #print(x1,y1,m1)
                pass
            #print(file_output)
            
            #returns scores/targets 
            #combo_move = number_gen(possible_range, list(x), list(mut_tag))
            
            cluster_center_sums = []
            all_cluster_centers = []
            all_inertia = []
            cluster_sums=[]
            cluster_centers=[]
            all_sil=[]
            x_reshape = np.array(c_data).reshape(-1,1)
            lowest_value_highest_cluster = []
            highest_value_highest_cluster = []  
            if len(x_reshape) == 1:
                possible_explanations = [1]
            elif len(x_reshape) <= 6:
                possible_explanations = list(range(max(halotypes),len(x_reshape)))       
            else:
                possible_explanations = list(range(max(halotypes),6)) 
             
            #print(possible_explanations) 
            for num in possible_explanations: 
                combo_move = number_gen([num], list(x), list(mut_tag))
                initial = np.array([float(x) for x in combo_move[1][0]]).reshape(-1,1)
                #kmeans clustering
                init_cluster = np.array(x[-num:]).reshape(-1,1)
                kmeans = KMeans(n_clusters=num, init=initial, random_state=10).fit(x_reshape)
                centers = kmeans.cluster_centers_            
                flat_list = [item for sublist in centers for item in sublist]
                
                all_cluster_centers.append(flat_list)                    
              
                #here we find the smallest value in the "highest freq" cluster
                largest_cluster_center = max(flat_list)
                label_largest_cluster = flat_list.index(largest_cluster_center)
                smallest_value_largest_cluster = \
                    [v for v,l in zip(c_data, kmeans.labels_) if l == label_largest_cluster]
                lowest_value_highest_cluster.append(min(smallest_value_largest_cluster))
               
                #this would be the "less ambiguous" method of calling consensus 
                highest_value_highest_cluster.append(max(smallest_value_largest_cluster))
                 
                cluster_center_sums.append(sum(flat_list))
                try: 
                    all_sil.append(silhouette_score(x_reshape, kmeans.labels_))
                except:
                    all_sil.append(0)
                    
            #now we have negative files in place
            if len(all_sil) != 0:        
                best_fit = max(all_sil)
                loc = all_sil.index(best_fit)
                best_fit = min(cluster_center_sums, key=lambda cluster_center_sums : abs(cluster_center_sums -1))
                loc=cluster_center_sums.index(best_fit)
                cluster_centers=all_cluster_centers[loc]
                cluster_sums=cluster_center_sums[loc]
                cluster_opt = possible_explanations[loc]
                possible_threshold_low = lowest_value_highest_cluster[loc]
                possible_threshold_high = highest_value_highest_cluster[loc]
                act_sil = all_sil[loc]
                mut_certainty = sum(y)
               
                cluster_centers.sort(reverse=True)
                try:
                    possible_threshold_amb = possible_threshold_high+0.015
                    possible_threshold = possible_threshold_low-0.015
                except:
                    possible_threshold=0
                    possible_threshold_amb=0
                #okay, lets ask what thresholds we could set k means method
                #no threshold returned if we think we have 1 thing
                if cluster_opt == 1:
                    possible_threshold=0
                    possible_threshold_amb=0
                #if the largest thing is less than 50% we ignore it
                elif cluster_centers[0] < 0.5:
                    possible_threshold_amb=0
                    possible_threshold=0
                elif max(halotypes) > 3:
                    possible_threshold_amb=0
                    possible_threshold =0
                elif cluster_opt == 2 or cluster_opt == 3:
                    if abs(cluster_centers[0]-cluster_centers[1]) < 0.15:
                        possible_threshold_amb=0
                        possible_threshold=0
                max_halotype = max(halotypes)
                
                temp_dict[file_output.replace(".json","")]= {
                    'max_halpotypes':max_halotype,\
                     'cluster_centers': cluster_centers,\
                     'sil_opt_cluster':cluster_opt,\
                     'sil':act_sil,\
                     'unique_mut':str(zipped), 'mut_certainty':mut_certainty,\
                     'threshold':possible_threshold, "threshold_amb":possible_threshold_amb}
            else: 
                temp_dict[file_output.replace(".json","")]= {
                    'max_halpotypes':0,\
                    'cluster_centers': 0,\
                    'sil_opt_cluster':0,\
                    'sil':0,\
                    'unique_mut':0, 'mut_certainty':mut_certainty,\
                    'threshold':0, 'threshold_amb':0}
        
    df_outcome = pd.DataFrame(temp_dict).T
    df_outcome = df_outcome.reset_index()
    final_df = df_outcome.merge(meta_df, on='index', how='left')
    final_df.to_csv("snv_output.csv")

    

def random_vis_code():    
    #look at relationship between accuracy and sil score
    correct = []
    for index, row in final_df.iterrows():
        if row['abundance_len'] != row['sil_opt_cluster']:
            correct.append('incorrect cluster num')
        else:
            correct.append('correct cluster num')
    
    sil_score = final_df['sil'].tolist()
    sns.set_style("whitegrid")
    sns.violinplot(x=correct, y=sil_score)
    plt.ylabel("Sil Score")
    plt.xlabel("Correct/Incorrectly predicted # clusters")
    plt.title("Relationship between weighed score accuracy and Sil Score")
    plt.savefig("sil.png")
    plt.close()

    #look at the relationship between predicted cluster centers and actual relative freq
    all_dist = []
    all_dist_weighed=[]
    for index, row in final_df.iterrows():
        predicted_centers = row['cluster_centers']
        actual_centers = ast.literal_eval(row['abundance'])
        predicted_centers.sort(reverse=True)
        actual_centers.sort(reverse=True)
        predict_len = len(predicted_centers)
        actual_len = len(actual_centers)
        if predict_len > actual_len:
            pad_num = predict_len - actual_len
            actual_centers.extend([0.0]*pad_num)
        elif actual_len > predict_len:
            pad_num = actual_len - predict_len
            predicted_centers.extend([0.0]*pad_num)
         

        from scipy.spatial import distance
        dist = 1-cosine(actual_centers, predicted_centers) 
        all_dist.append(dist)

  
 
    #distribution of distances from cluster centers
    sns.set_style("whitegrid")
    sns.distplot(x=all_dist, color='orange')
    sns.distplot(x=all_dist_weighed, color='purple')
    plt.xlabel("distance between GT freq and clsuter centers")
    plt.title("similarities between GT freq and cluster centers")
    plt.legend({'kmeans centers':'orange'})
    plt.savefig("dist.png")
    plt.close()


    sns.set_style("whitegrid")
    sns.scatterplot(x=all_dist, y=sil_score)
    plt.ylabel("Sil Score")
    plt.xlabel("Cosine similarity between predicted centers and actual frequencies")
    plt.title("Relationship between Sil score and freq prediction accuracy")
    plt.savefig("scatter.png")
    plt.close()


    #here we plot our "thresholds" against the % breakdown group
    sns.set_style("whitegrid")
    thresh_plot = [float(x) for x in final_df['threshold'].tolist()]
    gt_plot = [str(x) for x in final_df['abundance'].tolist()]
    for i,x in enumerate(gt_plot):
        if '3.33' in x:
            gt_plot[i]='33%'
        if '25.0' in x:
            gt_plot[i]='25%'
        if '20.0, 20.0' in x:
            gt_plot[i]='20%'
    g=sns.boxplot(x=gt_plot, y=thresh_plot)
    sns.swarmplot(x=gt_plot, y=thresh_plot, color='black')    
    plt.ylabel("Threshold")
    plt.xlabel("Ground Truth Abundance Group")
    plt.title("Predicted thresholds vs. ground truth frequency values")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("thresholds.png")
    plt.close()

    correct = 0
    incorrect = 0
    gt_values = []
    predicted = []
    #print some summary stats
    for index,row in final_df.iterrows():
        #print(index,row)
        if row['sil_opt_cluster'] == row['abundance_len']:
            correct += 1
        else:
            incorrect += 1
            gt_values.append(row['abundance'])
            predicted.append(row['cluster_centers'])

   
    #gives us the plot of actual num poplulations versus either
    #(1) haplotypes max
    #(2) snv max
    #(3) cluster max
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    sns.swarmplot(x='abundance_len',y='max_halpotypes', data=final_df)
    ax.axline((-1, 0), slope=1., color='red')
    #ax.set_xlim(0, 5)
    ax.set_ylim(0, 10) 
    #plt.ylim(0, 5)
    plt.xlabel("Number of Populations")
    plt.ylabel("Max num haplotypes")
    plt.title("Max haplotypes versus GT Populations") # You can comment this line out if you don't need title
    plt.savefig("snv_max_real.png")
    
def number_gen(values, freq, muts):
    """
    values : list
        List of a range of  number of frequencies we're looking for.
    freq : list
        possible frequencies
    muts : list
        list of all mutations
    """ 
    import itertools
    return_scores = []
    return_coms = []
    for target in values:
        
        all_scores = []
        all_coms = []
        combos = itertools.combinations(freq, target)
        mut_length = len([item for sublist in muts for item in sublist])
        for com in combos:    
            mut_locs = [freq.index(i) for i in list(com)]\
            #gather all muts for these freq
            mut_counts = [muts[i] for i in mut_locs]
            mut_counts = len([item for sublist in mut_counts for item in sublist])
            #add up all frequnecies
            summation = sum(com)
            #calculate closeness to 1
            dist = 1-abs(summation-1)
            #normalize and sum average mutations accounted for
            per_muts = (mut_counts/mut_length)/len(com)
            #weight these two things equally
            scores = (dist+per_muts)/2  
            all_scores.append(scores)
            all_coms.append(list(com))
        best = max(all_scores)
        loc = all_scores.index(best)
        best_com = all_coms[loc]
        return_scores.append(best)
        return_coms.append(best_com)
    return(return_scores, return_coms)

def call_consensus(filename, output_filename, threshold):
    """
    Given an input file, an ouput path, and a threshold, call consensus on a file.
    """ 
    cmd = "samtools mpileup -A -d 0 -Q 0 %s | ivar consensus -p %s -t %s" %(filename, output_filename, threshold) 
    os.system(cmd) 
   
def call_pangolin(multifasta, output_filename):
    """
    Given a multi-fasta and output filename, align and output lineages.
    """
    cmd = "pangolin %s --outfile %s --alignment" %(multifasta, output_filename)
    os.system(cmd)
 
def analyze_pangolin_results():
    """
    Analyze TP/FP/TN/FN for different consensus calls.
    """
    #hard coding the aa mutations of each lineage
    s_gene_dict = {"alpha":[69,70,144,501,570,614,681,716,982,1118], \
        "beta":[80,215,241,243,417,484,501,614,701], \
        "delta":[19,156,157,158,452,478,614,681,950], \
        "gamma":[18,20,26,138,190,417,484,501,614,655,1027,1176]}
 
    #open pangolin csv and parse out the S gene mutations

    #scatter plot TP/FP S gene mutations Y axis
    #consensus threshold X axis
    #color reflects automated versus non-automated
    


if __name__ == "__main__":
    main()
