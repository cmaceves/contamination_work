"""
Script takes in bam files and calculates per amplicon stats.
"""
import os
import sys
import ast
import copy
import json
import pysam
import itertools
import warnings
import argparse
import statistics
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from scipy.spatial import distance
import matplotlib.patches as mpatches
from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from util import calculate_positional_depths, create_regression_plot, create_barplot, create_boxplot
from call_external_tools import call_variants, call_getmasked, \
    retrim_bam_files, call_consensus

def warn(*args, **kwargs):
    pass

warnings.warn = warn
TEST=False
PRIMER_FILE = "/home/chrissy/Desktop/sarscov2_v2_primers.bed"
GT_STRAINS = {'alpha': 'B.1.1.7', 'beta':'B.1.351', 'gamma':'P.1', 'delta':'B.1.617.2'}
metadata_path = '../spike-in_bams_spikein_metadata.csv'

def extract_amplicons(bam, primer_0, primer_1, primer_0_inner, primer_1_inner, \
        noise_dict, pos_dict, primer_drops=None):
    """
    Parameters
    -----------
    bam : str
        Path to the bam file
    primer_0 : int
        The star pos of the forward primer
    primer_1 : int
        The end pos of the reverse primer

    Returns
    -------

    Function takes in bam file locations, primer locations, and nucleotides to ignore by position and 
    returns the relative propotion of amplicon level haplotypes observed and positions in which mutations
    occur. 
    """
    #if TEST:
    print("extracting amplicons for ", bam, primer_0)

    #adjust for ends of possible reads
    primer_0 -= 1
    primer_1 += 1 
   
    #encoded nucs, 0 means match ref
    encoded_nucs = {"A":1,"C":2,"G":3,"T":4,"N":5, "D":6, "SC":-1, "M":0}

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
 
    #amp len CHANGED
    amp_len = primer_1 - primer_0
    amp_indexes = list(range(primer_0, primer_1))
    
    #make sure we don't get a pair twice
    seen_reads = []
    total_count = 0

    #create pileup
    for pileupcolumn in samfile.pileup("NC_045512.2", start=primer_0, stop=primer_1):
        for pileupread in pileupcolumn.pileups:
            #-1 denote soft clipping or not observed
            amp = [-1]*amp_len
            freq = [-1]*amp_len
            quality = [0.0] *amp_len
            qualifies = False

            #determine the read direction
            reverse = pileupread.alignment.is_reverse
            if reverse:
                primer_direction = 'R'
                if  pileupread.alignment.reference_end -10 < primer_1_inner < pileupread.alignment.reference_end + 10:
                    qualifies = True
            else:
                primer_direction = 'F'
                if  pileupread.alignment.reference_start -10 < primer_0_inner < pileupread.alignment.reference_start + 10:
                    qualifies = True            
            
            #ignore if we're seen this read before
            if pileupread.alignment.qname+"_"+primer_direction in seen_reads:
                continue

            #read belongs to amplicon we're looking at
            if qualifies:
                seen_reads.append(pileupread.alignment.qname+"_"+primer_direction)                 
                ref_seq = list(pileupread.alignment.get_reference_sequence())
                total_ref = pileupread.alignment.get_reference_positions(full_length=True)
                total_query = list(pileupread.alignment.query_sequence)
                total_qualities = list(pileupread.alignment.query_qualities)
                alignment_object = pileupread.alignment.get_aligned_pairs(with_seq=True)
                for count, tr in enumerate(total_ref):
                    if tr is None:
                        ref_seq.insert(count, None)

                cigar = pileupread.alignment.cigartuples
                cigar = [list(x) for x in cigar]
                expand_cigar = []
                for temp_cig in cigar:
                    expand_cigar.extend([temp_cig[0]]*temp_cig[1]) 
                
                for count, cig in enumerate(expand_cigar):
                    if cig == 2:
                        total_query.insert(count, None)
                        total_ref.insert(count, None)
                        total_qualities.insert(count, None)
                        ref_seq.insert(count, None)

                #used to track blocks of deletions and insertions
                on_insertion=False
                on_deletion=False
                deletion = ''
                insertion = ''
            
                for pcount,(pos,nuc,qual,cigtype, al, rs) in enumerate(zip(total_ref, total_query, total_qualities, \
                        expand_cigar, alignment_object, ref_seq)):
                    if pos is None and cigtype == 2:
                        on_deletion = True
                        deletion += al[2].upper()
                        continue
                    elif pos not in amp_indexes:
                        continue    
                    nuc = nuc.upper()

                    #if we have an insertion we create a seperate encoding for it, and place it in    
                    if cigtype == 1:
                        on_insertion = True
                        insertion += nuc.upper()
                        continue
                    if cigtype == 4 or cigtype == 5:
                        continue
                    elif cigtype == 0:
                        if on_insertion:
                           pos = pos - len(insertion)
                           nuc = '+'+insertion
                           insertion=''
                           ref = None
                           on_insertion=False
                        elif on_deletion:
                            pos = pos - len(deletion)
                            
                            nuc = '-'+deletion
                            deletion=''
                            ref = None
                            on_deletion=False
                            
                        else:
                            nuc = nuc.upper()
                            ref = rs.upper()
                   
                        if nuc not in encoded_nucs:
                            new_value = max(encoded_nucs, key=encoded_nucs.get) 
                            encoded_nucs[nuc] = encoded_nucs[new_value]+1
                        
                    #double check that it's on our amplicon 
                    if pos in amp_indexes:
                        #this means it either it's a low level mutation or matches reference
                        if int(pos) in noise_dict:
                            noise = noise_dict[int(pos)]
                            if nuc in noise or nuc == al[2]:                           
                                loc = amp_indexes.index(pos)
                                amp[loc] = 0.0
                                freq[loc] = 0.0
                                continue
                        loc = amp_indexes.index(pos)
                        amp[loc] = encoded_nucs[nuc]
                        quality[loc] += qual   
                        try:
                            #then we go find the overall freq
                            temp = pos_dict[str(pos)]['allele'][nuc]['count']/pos_dict[str(pos)]['total_depth']
                            freq[loc]=temp
                        except:
                            pass
                            #print("failed to find " + str(pos) + " in " + bam + " pos dict")
                           
                found_amps += 1           
 
                #once we've added all the pos to the read array, we append to the matrix
                read_matrix.append(amp)
                freq_matrix.append(freq)
                quality_matrix.append(quality)

    read_matrix=np.array(read_matrix)
    freq_matrix=np.array(freq_matrix)
    quality_matrix=np.array(quality_matrix)

    poi = []
    relative_poi=[]
    mutation_nt = []

    #invert the dictionary key/value pairs
    inv_map = {v: k for k, v in encoded_nucs.items()}

    #iterate by position
    for c, (pos_array, qual_array) in enumerate(zip(read_matrix.T, quality_matrix.T)):
        gt_pos = c+primer_0
 
        #positions that match the reference
        match_ref = pos_array[pos_array==0]
        #positions that are soft clipped
        soft_clipped = pos_array[pos_array==-1]
        #positions that have mutations
        mutations = pos_array[pos_array>0]
        #if we don't have mutations here we don't care
        if len(mutations) == 0:
            continue
        
        #remover blank values from quality
        filter_qual = qual_array[qual_array>0]
        #find the average quality here
        avg_qual = np.average(filter_qual)
        if avg_qual < 20:
            continue        

        #depth doesn't include the softclipped
        total_depth = len(match_ref) + len(mutations)
        if total_depth < 10:
            continue

        if TEST: 
            print("pos", primer_0+c, \
                "soft clipped", len(soft_clipped), \
                "match ref", len(match_ref) , \
                "mutations", len(mutations))
        

        #decode the mutations 
        values, counts = np.unique(mutations, return_counts=True)
        percent = [x/total_depth for x in counts]
 
        relative_poi.append(c)
        poi.append(c+primer_0)

    if read_matrix.ndim < 2:
        read_matrix = np.zeros((0,0))  
   
    max_groups = []
    group_counts=[]
    groups = []
    groups2 = []
    mut_groups = []
    poi_order=[]

    if len(relative_poi) > 0:
        filter_matrix = freq_matrix.T[relative_poi]
        read_matrix_filt = read_matrix.T[relative_poi]
        quality_matrix_filt = quality_matrix.T[relative_poi]

        if filter_matrix.shape != read_matrix_filt.shape != quality_matrix_filt.shape:
            print("error in matrix shape.")
            sys.exit(1)

        #total used
        cc = 0
        
        for count, (thing,thing2) in enumerate(zip(filter_matrix.T, read_matrix_filt.T)):
            if -1 in thing2 or -1 in thing:
                continue 
            #counts toward our total haplotype depth
            cc += 1

            #if it only matches the reference we ignore it
            muts_contained = list(np.unique(thing2))
            if len(muts_contained) == 1 and muts_contained[0] == 0:
                continue

            decoded_mutations = [inv_map[a] for a in thing2]
            thing = [str(x) for x in list(thing)]
            stringify = '-'.join(thing)
            thing = [float(x) for x in thing]
            if stringify not in groups:
                groups.append(stringify)
                groups2.append(list(thing))
                mut_groups.append(decoded_mutations)
                group_counts.append(1)
            else:
                loc = groups.index(stringify)
                group_counts[int(loc)] += 1
    
    #print(group_counts, groups, cc)
    group_percents = [x/cc for x in group_counts]   
     
    """
    #remove low occuring groups
    positions_to_remove = []
    for i, perc in enumerate(group_percents):
        if perc < 0.03 or perc > 0.97:
            positions_to_remove.append(i)
            del group_percents[i]
            del groups[i]
            del groups2[i]
            del mut_groups[i]
    
    group_percents = [a for i,a in enumerate(group_percents) if i not in positions_to_remove]
    groups = [a for i,a in enumerate(groups) if i not in positions_to_remove]
    groups2 = [a for i,a in enumerate(groups2) if i not in positions_to_remove]
    mut_groups = [a for i,a in enumerate(mut_groups) if i not in positions_to_remove]
    """

    #if we've managed to eliminate all haplotypes, remove poi too
    if len(group_percents) == 0:
        poi = []

    #let's make sure that we don't have a strange haplotype split
    #check to make sure that the haplotypes are sufficiently different if 
    #our mutations for this amplicon exceed 2
    removal = []
    if len(poi) > 2:    
        add_to = [] 
        add_val = []
        for i,(hp,mg,g2)  in enumerate(zip(group_percents, mut_groups, groups2)):  
            #if the haplotype is less than 10% we consider it for merge
            if hp < 0.10:
                for i2, g3 in enumerate(groups2):
                    if i in removal:
                        continue    
                    #we're looking at the same group
                    if g3 == g2:
                        continue
                    #cosine similarity
                    sim_score= 1 - cosine(g3,g2)
                    if sim_score > 0.7:
                        print(
                            "groups2", groups2, "\n",\
                            "add", g2, " to ", g3, "\n"
                            )
                        #this is matching with a group that's already been collapsed
                        if i2 in add_to or i in removal:
                            loc = add_to.index(i2)
                            removal.append(i)
                            add_to.append(add_to[loc])
                            add_val.append(hp)
                        else:
                            removal.append(i)
                            add_to.append(i2)
                            add_val.append(hp)
    
    removal.sort(reverse=True)
    
    if len(removal) > 0:
        for x,y in zip(add_to, add_val):
            group_percents[x] += y
        print(group_percents)
        for i in removal:
            del group_percents[i]
            del mut_groups[i]
            del groups2[i]
           
    if TEST:
        print("primer 0", primer_0, "\n", \
                "poi", poi, "\n",\
                "haplotype percents", group_percents, "\n", \
                "mutation groups", mut_groups, "\n", \
                "groups2", groups2
                )
        sys.exit(0)
    return(poi, group_percents, found_amps, groups2, mut_groups)


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


    primer_pairs = {}
    #iterate over the primer list again, this time save outer pos info
    for pl in content:
        pn = pl[3][:-1]
        if pn in primer_pairs:
            pass
        else:
            primer_pairs[pn] = [0,0]
        
        direct = pl[3][-1]
        if direct == "F":
            val = pl[1]
            primer_dict[pn][0] = val
            primer_dict_inner[pn][0] = pl[2]
            primer_pairs[pn][0] = pl[-1]
        if direct == "R":
            val = pl[2]
            primer_dict[pn][1] = val
            primer_dict_inner[pn][1] = pl[1]
            primer_pairs[pn][1] = pl[-1]

    return(primer_dict, primer_dict_inner)


def parallel(primer_dict, bam, total_pos_depths, noise_dict, primer_dict_inner, \
        basename, n_jobs):
    """
    Function takes in primer dict and bam and parallel processes each file to 
    extract amplicon information.
    """
    results = Parallel(n_jobs=n_jobs)(delayed(extract_amp_parallel_wrapper)(k, v, bam, \
            total_pos_depths, noise_dict, \
            primer_dict_inner) for k,v in primer_dict.items())
    file_level_dict = {}
    for r in results:
        file_level_dict[r[0]] = r[1]

    if TEST:
        print(file_level_dict)
 
    with open("../json/%s.json" %basename, "w") as jfile:
        json.dump(file_level_dict, jfile)
    
    print(basename, "finished processing")

def parse_snv_output(csv_filename):
    """
    Parameters
    ----------
    csv_filename : str
        Full path the location of the .csv file with clustering output.

    Gets filenames and thresholds from the .csv file output.
    Returns dict with filenames and thresholds.
    """

    file_threshold_dict = {}
    df = pd.read_csv(csv_filename)
    for index, row in df.iterrows():
        thresh_low = row['threshold_low'] + 0.10
        filename = row['index']
        thresh_high = row['threshold_high'] - 0.10
        bam_filename = "../simulated_data/final_simulated_data/" + filename + ".bam"
        file_threshold_dict[bam_filename] = {"threshold_low": round(thresh_low,2), "output_low": "../consensus/low_10/"+\
            filename +"_"+ str(round(thresh_low,2)), 'threshold_high':round(thresh_high,2), "output_high": \
            "../consensus/high_10/"+filename+"_"+str(round(thresh_high,2))}
    return(file_threshold_dict)


def retrim_bam_files(all_filenames, output_dir):
    """
    Parameters
    ----------
    all_filenames : list
        List of full paths to retrim.
    output_dir : str
        Directory to output files to.

    Function to take the original .bam files in and retrim them for consistency.
    """
    for filename in all_filenames:
        basename = filename.split("/")[-1].split("_s")[0]
        if os.path.isfile(os.path.join(output_dir, basename+".final.bam")):
            continue
        final_bam = retrim_bam_files(filename, basename, output_dir, ref_seq, primer_bed)
        if TEST:
            print(final_bam)


def main():
    DATA_TYPE = "simulated"
    REF_SEQ = "/home/chrissy/Desktop/sequence.fasta"
    RETRIM_BAMS = False
    PROCESS_DATA = True
    POST_PROCESS = False
    VISUALIZE = False
    CONSENSUS = False
    NEXTSTRAIN_ANALYSIS = False
    METRIC = "bic" 
    n_jobs = 7
 
    if DATA_TYPE == "wastewater":
        datapath = "/home/chrissy/Desktop/retrimmed_bam" 
    elif DATA_TYPE == "simulated":
        datapath = "/home/chrissy/Desktop/simulated_data/final_simulated_data"

    if PROCESS_DATA is True and RETRIM_BAMS is False:
        all_filenames = [x for x in os.listdir(datapath) if x.endswith('.bam')]
        all_filenames = [os.path.join(datapath, x) for x in all_filenames]
        primer_dict, primer_dict_inner  = get_primers(PRIMER_FILE) 
      
    if PROCESS_DATA is True:     
        for filename in all_filenames:
            process(filename, n_jobs)

    if POST_PROCESS is True:
        #creates the .csv file with thresholds and other info
        group_files_analysis(file_folder, primer_dict, output_dir)

    if VISUALIZE is True:
        snv_df = pd.read_csv("snv_output.csv")
        simulated_metadata = pd.read_csv("../simulated_metadata.csv")
        wastewater_metadata = pd.read_csv(metadata_path)
        wastewater_metadata = reformat_metadata(wastewater_metadata)
        create_regression_plot(snv_df, wastewater_metadata)    
        create_boxplot(snv_df,  wastewater_metadata)

    if CONSENSUS is True:
        #parses the tsv containing thresholding information
        file_threshold_dict = parse_snv_output("snv_output.csv") 

        #remove old specific consensus thresholds
        con = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
       
        #calls consensus using the above parsed thresholds
        for k, v in file_threshold_dict.items():
            #print("Calling consensus on ", k)
            thresh_low = v['threshold_low']
            output_low = v['output_low']
            thresh_high = v['threshold_high']
            output_high =v['output_high']
             
            if os.path.isfile(output_low + ".fa"):
                continue
                #if float(threshold) > 0.50:
            call_consensus(k, output_low, thresh_low)            
            if os.path.isfile(output_high + ".fa"):
                continue
                #if float(other_thresh) > 0.5:
            call_consensus(k, output_high, thresh_high)
         
        """     
        #calls consensus using an array of thresholds 0.5-0.95 every 0.05     
        possible_consensus = [round(x,2) for x in list(np.arange(0.5,1.00,0.05))]
        for k,v in file_threshold_dict.items():
            for pc in possible_consensus:
                new_output = k.split("/")[-1].split(".")[0]
                new_output = "../consensus/steady/" + new_output + "_" + str(pc)           
                if os.path.isfile(new_output + ".fa"):
                    continue
                call_consensus(k, new_output, pc)

        sys.exit(0)
        """

    if NEXTSTRAIN_ANALYSIS is True:
        metadata_df = pd.read_csv("snv_output.csv")
        #lower_consensus_scramble(metadata_df)
        threshold_variation_plot("../nextclade_plus_minus.csv", metadata_df)    
        analyze_nextstrain("../low_nextclade.csv", "../high_nextclade.csv", metadata_df)


def threshold_variation_plot(nextclade_filename, metadata):
    """
    Plots the results of varying the consensus threshold.
    """
    nextclade_results = pd.read_table(nextclade_filename, sep=';')

    #color
    filetype_list = []
    #percent varied from threshold
    percent_varied = []
    #percent mutations missing from upper population
    percent_mut_missing = []
    #percent mutations from lower population
    percent_mut_extra = []

    mutations_table = "../key_mutations.csv"
    df = pd.read_csv(mutations_table)
    strain_muts = {'alpha':[], 'beta':[], 'delta':[], 'gamma':[]}
    for index, row in df.iterrows():    
        mut = row['gene'] + ":" + row['amino acid']
        strain_muts[row['lineage']].append(mut)

    test = 'simulated_alpha_delta_60_40' 
    for index, row in nextclade_results.iterrows():
        nextstrain_filename = row['seqName']
        metadata_filename = "_".join(nextstrain_filename.split("_")[1:-6])        
        if metadata_filename != test:
            continue
        print(metadata_filename)
        direction = nextstrain_filename.split("_")[6]
        variance= nextstrain_filename.split("_")[7]
        gt_list = metadata_filename.split("_")        
        if int(gt_list[-1]) > int(gt_list[-2]):
            upper_strain = gt_list[-3]
            filetype = gt_list[-2]+"/"+gt_list[-1]
            lower_strain = gt_list[1]
        else:
            upper_strain = gt_list[1]
            lower_strain = gt_list[-3]
            filetype = gt_list[-1]+"/"+gt_list[-2]
               
        #expcted muts
        #expected_muts = strain_muts[upper_strain]
        #unexpected_muts = strain_muts[lower_strain]
        expected_muts = get_mutations_vcf(upper_strain)
        unexpected_muts = get_mutations_vcf(lower_strain)
        #print(lower_strain, upper_strain)
        #muts detected via nextclade
        
        muts = []
        if str(row['substitutions']) != 'nan':
            muts = [int(x[1:-1]) for x in row['substitutions'].split(",")]
        if str(row['deletions']) != 'nan':
            muts.extend(row['deletions'].split(","))
        if str(row['insertions']) != 'nan':
            muts.extend(row['insertions'].split(","))
        
        if direction == 'minus':
            continue
            percent_varied.append(-1*float(variance))
        else:
            percent_varied.append(float(variance))

        filetype_list.append(str(filetype))
        percent_recovered = len([x for x in muts if x in expected_muts])/len(expected_muts)
        percent_mut_missing.append(percent_recovered)
    
    
    sns.set_style("whitegrid")
    sns.scatterplot(x=percent_varied, y=percent_mut_missing, hue=filetype_list)        
    plt.savefig("./figures/scramble.png")
 
def lower_consensus_scramble(metadata):
    """
    Takes the files where we think we can capture the upper population and varies the 
    consensus threshold by 10 percent in +/- direction in 1 percent increments.
    """
    vary_threshold = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10]
    for index, row in metadata.iterrows():
        sil_score = float(row['sil'])
        lower_thresh = float(row['threshold_low'])
        
        #this is file can call consensus to capture the upper population
        if sil_score > 0.75 and lower_thresh > 0.5:
            for delta_thresh in vary_threshold:
                output_file_one = "../consensus/plus_minus/"+row['index']+"_minus_"+str(delta_thresh)
                output_file_two = "../consensus/plus_minus/"+row['index']+"_plus_"+str(delta_thresh)
                thresh_one = lower_thresh - delta_thresh
                thresh_two = lower_thresh + delta_thresh
                
                filename = "../simulated_data/final_simulated_data/"+row['index']+".bam"
                call_consensus(filename, output_file_one, thresh_one)    
                call_consensus(filename, output_file_two, thresh_two)
        
def analyze_nextstrain(filename, filename_2, metadata):
    """
    Parameters
    ----------
    filename : str 
        Full path to the .tsv with nextclade output.
    metadata : pd.DataFrame
        Dataframe containing information to what is contained within the samples.    

    Parse apart nextclade file and analyze the consensus calling data.
    """
    
    low_df = pd.read_table(filename, sep=';')
    high_df = pd.read_table(filename_2, sep=';')
    
     
    #plot where we chose to call consensus
    sil_scores = metadata['sil']
    filenames_meta = metadata['index'].tolist()
    threshold_low = metadata['threshold_low'].tolist()
    threshold_high = metadata['threshold_high'].tolist() 

    order_list = ['50/50', '45/55', '40/60', \
              '35/65', '30/70', '25/75', '20/80', \
              '15/85', '10/90', '05/95', '0/100']    
    filetype_list = copy.deepcopy(order_list)
    filetype_list.extend(filetype_list)
    filetype_list.sort()
    
    df_dict = {"filetype": filetype_list , "Attempt to Recover\nUpper Population": ['yes','no']*len(order_list), \
        'count':[0]*len(filetype_list)}
     
    for index, row in metadata.iterrows():
        gt_list = row['index'].split("_")                
        if int(gt_list[-1]) > int(gt_list[-2]):
            filetype = gt_list[-2]+"/"+gt_list[-1]
        else:
            filetype = gt_list[-1]+"/"+gt_list[-2]
         
        loc = df_dict['filetype'].index(filetype)
        sil_score = float(row['sil'])
        lower_thresh = float(row['threshold_low'])
        
        if sil_score > 0.75 and lower_thresh > 0.5:
            df_dict['count'][loc] += 1
        else:
            loc += 1
            df_dict['count'][loc] += 1

    norm_list = []
    base_val = 0
    for i, item in enumerate(df_dict['count']):
        if i % 2 == 1:
            base_val += item
            norm_list.append(base_val)
            norm_list.append(base_val)
            base_val = 0
        else:
            base_val += item
    df_dict['count'] = [x/y for x,y in zip(df_dict['count'], norm_list)]
    
    create_barplot(pd.DataFrame(df_dict), "filetype", "count", "Attempt to Recover\nUpper Population")
    
    
    #here we ask how many mutations are recovered from each of the mutations    

    sys.exit(0)
    
    #did we recover the correct lineage? 
    df_dict = {"filetype":[], "recovered": [], "not_recovered":[], "total":[]}

    #did we recover the correct lineage?
    for index, row in low_df.iterrows():
        nextstrain_filename = row['seqName']
        nextstrain_pango = row['Nextclade_pango']
        metadata_filename = "_".join(nextstrain_filename.split("_")[1:-5])
        metadata_row = metadata[metadata['index'] == metadata_filename]

        gt_list = metadata_filename.split("_")        
        if int(gt_list[-1]) > int(gt_list[-2]):
            upper_strain = gt_list[-3]
            filetype = gt_list[-2]+"/"+gt_list[-1]
        else:
            upper_strain = gt_list[1]
            filetype = gt_list[-1]+"/"+gt_list[-2]
        
        sil_score = float(metadata_row['sil'])
        lower_thresh = float(metadata_row['threshold_low'])
        if sil_score > 0.75 and lower_thresh > 0.5:
            exact_strain = GT_STRAINS[upper_strain] 
            if nextstrain_pango == exact_strain:
                recovered = True
            else:
                recovered = False
            if filetype in df_dict['filetype']:
                loc = df_dict['filetype'].index(filetype)
                df_dict['total'][loc] += 1
                if recovered is True:
                    df_dict['recovered'][loc] += 1
                else:
                    df_dict['not_recovered'][loc] += 1                 
            else:
                df_dict['filetype'].append(filetype)
                df_dict['total'].append(1)
                if recovered is True:
                    df_dict['recovered'].append(1)
                    df_dict['not_recovered'].append(0)
                else:
                    df_dict['recovered'].append(0)
                    df_dict['not_recovered'].append(1)

    df_dict['recovered'] = [x/y for x,y in zip(df_dict['recovered'], df_dict['total'])]
    df_dict['not_recovered'] = [x/y for x,y in zip(df_dict['not_recovered'], df_dict['total'])]
    create_barplot(pd.DataFrame(df_dict)) 
    sys.exit(0)

   
    #plots varation in mutations found in technical replicates, and predicted thresholds
    #variation_in_technical_replicates(metadata, df)
    sys.exit(0)
    global_tpr = []
    global_fpr = []
    global_percents = []
    global_strains = []
    global_thresh = []
    gamb = []
    gtamb = []
    global_sil = []
    global_color = []

    other_sil=[]
    other_clade=[]

    file_type=[]
    consensus_threshold=[]
    correct_clade=[]
    lower_upper=[]
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
        sil_current = sil_dict[fileName]['sil']
        thresh_current = sil_dict[fileName]['thresh']
        #ground truth threshold of consensus run
        act_thresh = float(seqName.split("_")[3])
        #ambuguity threshold
        amb_thresh = round(sil_dict[fileName]['thresh_amb'],2)
        
        color=''
        if round(act_thresh,2) == round(thresh_current,2):
            color='conserve/lower'
            #print(color, act_thresh, thresh_current)
        elif round(act_thresh,2) == round(amb_thresh,2):
            color='amb/upper'
            
        else:
            continue
        
        #print(round(sil_current,2), act_thresh, thresh_current, amb_thresh)
        
        meta_row = metadata.loc[metadata['index']==fileName]
        try:
            variants = ast.literal_eval(meta_row['variant'].tolist()[0])
        except:
            continue
        if act_thresh == 0:
            continue
        variants = [x.lower() for x in variants]
        relative_percent = ast.literal_eval(meta_row['abundance'].tolist()[0])
        #print(variants, nextclade_pango, clade, act_thresh, relative_percent)
        clade_act = clade.split("(")[-1].split(',')[0].lower().replace(")","")
        if nextclade_pango == 'A':
            clade_act = 'aaron'
        if clade_act == variants[-1]:
            correct_clade.append('yes')
        else:
            print(variants, nextclade_pango, clade_act, act_thresh, relative_percent, color)
            correct_clade.append('no')
        if len(relative_percent) == 1:
            file_type.append(str(relative_percent[0]))
        elif len(relative_percent) == 5:
            file_type.append(str(20))
        elif len(relative_percent) == 3:
            file_type.append(str(33))
        elif len(relative_percent) == 4:
            file_type.append(str(20))
        elif 95 in relative_percent:
            file_type.append("95/5")
        elif 90 in relative_percent:
            file_type.append("90/10")
        elif 80 in relative_percent:
            file_type.append("80/20")
        elif 60 in relative_percent:
            file_type.append("60/40")
        elif 50 in relative_percent:
            file_type.append("50/50")
        consensus_threshold.append(act_thresh)
        other_sil.append(sil_current)
        lower_upper.append(color)
        continue
        #if 60 not in relative_percent:
        #    continue
        
        #if aaron is the largest value
        #if variants[1] == 'aaron':
        #    continue
        
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
        #if act_thresh == 0.5:
        #    pass
        #else:
        #    continue
        #print(act_thresh, relative_percent, variants, color, thresh_current, amb_thresh, seqName)
        #sys.exit(0)
        """
        print(act_thresh)
        print(actual_strain)
        print(found_muts)
        print(muts_to_see)
        print(relative_percent)
        """
        #muts to see are expected S gene mutations
        #found_muts are muts found in this sample
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
            #print(muts_to_see, found_muts, variants, act_thresh, relative_percent) 
            global_tpr.append(tp/(tp+fn))
            global_fpr.append(fp/(tp+fn))
            global_percents.append(str(variants))
            global_thresh.append(act_thresh)
        #print(global_thresh)
        #print(np.max(global_thresh), np.max(global_tpr))
        global_sil.append(sil_current)
        global_color.append(color)
        #print(act_thresh, relative_percent, variants, color, thresh_current, amb_thresh, seqName)
        #sys.exit(0)

    yep = list(np.unique(file_type))
    yep.extend(yep)
    yep.sort()
    dict_temp = {'ft':yep, 'count':[0.0]*len(yep), "type":['lower','upper']*(int(len(yep)/2))}
    print(dict_temp)
    other_yep =list(np.unique(file_type))
    for ft, lu in zip(file_type, lower_upper):
        loc = other_yep.index(ft)
        loc = loc*2
        if 'lower' in lu:
            loc = loc+1
        dict_temp['count'][loc] += 1
    df_t = pd.DataFrame(dict_temp)
    print(df_t)
    sns.barplot(x="ft", y="count", hue='type', data=df_t)
    plt.savefig('bar.png')
    sys.exit(0)
    
    sns.boxplot(x=lower_upper, y=other_sil, hue=correct_clade,\
                 linewidth=1)
    plt.xlabel("correct clade")
    plt.ylim(0,1)
    plt.ylabel("Sil Score")
    plt.title("Accuracy by sil score")
    plt.tight_layout() 
    plt.savefig("box.png")
    plt.close()
    sys.exit(0)
    sns.scatterplot(x=global_sil, y=global_tpr, hue=global_color, palette={"conserve/lower":"red", \
            "amb/upper":"blue"})
    plt.axvline(x=0.60, color='orange',linewidth=2 )
    plt.xlabel("sil score")
    plt.xlim(0 ,1)
    plt.ylim(0,1)
    plt.ylabel("true positive rate (expected S gene mutations)")
    plt.title("40/60 files")
    plt.tight_layout() 
    plt.savefig("40_60_scatter.png")
    plt.close()
    sys.exit(0)
    sns.jointplot(x=global_thresh, y=global_fpr, kind='hex', color='black', gridsize=15, \
        extent=[0, 1, 0, 1])
   
    plt.axvline(x=float(np.average(gtamb)), color='red', linewidth=2)
    plt.axvline(x=float(np.average(gamb)), color='blue', linewidth=2)
    plt.axvline(x=0.60, color='orange',linewidth=2 )
    #sns.scatterplot(x=global_thresh, y=global_tpr, hue=global_percents)
    plt.xlabel("consensus thresholds")
    plt.xlim(0 ,1)
    plt.ylim(0,1)
    plt.ylabel("false positive rate (expected S gene mutations)")
    plt.title("40/60 files")
    plt.tight_layout() 
    plt.savefig("40_60_test.png")
    plt.close()
    
def process(bam, n_jobs):
    """
    Parameters
    ----------
    bam : str
        Path to the bam of interest.
    n_jobs : int
        The number of jobs to concurrently run.
    """
    basename = bam.split("/")[-1].split(".")[0].replace("_sorted","")
    primer_file = "../sarscov2_v2_primers.bed"  
    reference_filepath = "../sequence.fasta"
    variants_output_dir = "../variants"
    bed_filepath = "../sarscov2_v2_primers.bed"
    primer_pair_filepath = "../primer_pairs.tsv"
    masked_output_dir = "../masked"
    
    if TEST is False:
        if os.path.isfile("../json/"+basename+".json"):
            return
    
    #this block calls get masked and variants
    variants_check = os.path.join(variants_output_dir, "variants_"+basename+".tsv")
    if not os.path.isfile(variants_check):
        #call the ivar variants and generate .tsv
        call_variants(bam, basename, reference_filepath, variants_output_dir)

    masked_check = os.path.join(masked_output_dir, "masked_"+basename+".txt")
    if not os.path.isfile(masked_check):
        #call getmasked to get the primer mismatches
        call_getmasked(bam, basename, variants_output_dir, bed_filepath, \
            primer_pair_filepath, masked_output_dir)
    
    
    primer_dict, primer_dict_inner  = get_primers(primer_file)
      
    if not os.path.isfile("../pos_depths/%s_pos_depths.json" %basename): 
        #try:
        total_pos_depths = calculate_positional_depths(bam)
        with open("../pos_depths/%s_pos_depths.json" %basename, "w") as jfile:
            json.dump(total_pos_depths, jfile) 
        #except:
        #    return(1)
    else:
        with open("../pos_depths/%s_pos_depths.json" %basename, "r") as jfile:
            total_pos_depths = json.load(jfile)

    #return(0) 
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
        #if we have a reference value
        if bool(v['ref']):
            ref = max(v['ref'], key=v['ref'].get)
            noise_dict[int(k)].append(ref)
        for nt, count in alleles.items():
            if count['count']/total_depth < 0.03:
                noise_dict[int(k)].append(nt)
     

    #TEST LINES
    if TEST:
        test_dict = {}
        for k,v in primer_dict.items():
            if int(v[0]) == 21474:
                test_dict[k] = v
        primer_dict = test_dict
        n_jobs = 1 

    p_0=[]
    p_1=[]
    p_0_inner=[]
    p_1_inner=[] 
    count = 0
    file_level_dict = {}
    parallel(primer_dict, bam, total_pos_depths, noise_dict, primer_dict_inner, \
            basename, n_jobs)  

def extract_amp_parallel_wrapper(k, v, bam, total_pos_depths, noise_dict,\
        primer_dict_inner):
    """
    Function takes in the primers and a bam file and calls the extract amplicons 
    function to recover amplicon level
    (1) haplotypes
    (2) haplotype frequencies
    (3) positions of mutations
    (4) mutation frequencies
    (5) the number of reads per amplicon
    """

    primer_0 = int(v[0])
    primer_1 = int(v[1])

    primer_0_inner = int(primer_dict_inner[k][0])
    primer_1_inner = int(primer_dict_inner[k][1])
    
    #we didn't successfully find a primer pair
    if primer_0 == 0.0 or primer_1 == 0.0 or primer_0_inner == 0 or primer_1_inner == 0:
        return(primer_0, {})
    poi, groups_percents, found_amps, freqs, mut_groups = extract_amplicons(bam, primer_0, primer_1, primer_0_inner, primer_1_inner, noise_dict, total_pos_depths)

    amplicon_level_dict = { "poi":poi, \
        "haplotype_percents": groups_percents, "found_amps": int(found_amps), \
        'mut_groups':mut_groups, 'poi_freq': freqs}
    
    return(primer_0, amplicon_level_dict)

def group_files_analysis(file_folder, primer_dict, bam_dir):  
    """
    Parameters
    ----------
    file_folder : str
        Location where the .json files containing amplicon level mutations are found.
    primer_dict : dict
        Dictionary relating primers names to their start position.
    bam_dir : str
        Full path to the directory containing the .bam files.

    Function takes in a directory, and iterates the directory processing all json files.
    Each json file contains amplicon level information that is used to cluster within
    each file.
    """

    print("Begining group file analysis")

    #load metadata
    metadata_path = '../spike-in_bams_spikein_metadata.csv'
    metadata_df = pd.read_csv(metadata_path)
   
    #get columns for downstream.... could be written better
    meta_filenames = [x.split('.')[0] for x in metadata_df['filename'].tolist()]
    meta_abundance = [ast.literal_eval(x) for x in metadata_df['abundance(%)'].tolist()]
    meta_abundance_count = [len(x) for x in meta_abundance]
    meta_df = pd.DataFrame({"index":meta_filenames, "abundance_len":meta_abundance_count, \
        'abundance':[str(x) for x in meta_abundance], "variant":metadata_df['variant'].tolist()})

    #get the json paths for the results
    json_filenames = [os.path.join(file_folder, x) for x in os.listdir(file_folder) if x.endswith(".json")]
    
    #where we store all results prior to writing to .csv 
    temp_dict = {}

    #iterate through every output result
    for file_output in json_filenames:
        #can't open what doesn't exist --__(..)__--
        if not os.path.isfile(file_output):
            continue
        
        basename = file_output.split("/")[-1].replace(".json","")
        
        #for this file also fetch this mismatched primers
        mismatched_primers = []        
        masked_loc = os.path.join("/home/chrissy/Desktop/masked", "masked_"+basename+".txt")
        with open(masked_loc, 'r') as mfile:
            for line in mfile:
                mismatched_primers.append(line.strip()[:-1])
        mismatched_primers = list(np.unique(mismatched_primers))             

        if TEST:
            print("eval", basename)    

        #all the haplotype frequencies in one large list
        total_haplotype_freq = []
        total_mutation_freq = []
        total_positions = []
        total_positions_extend = []
        thf_condensed = [] #total haplotype frequencies with amplicon grouping
        mf_condensed = []
        improper_primers = []
        total_amps = []

        with open(file_output, 'r') as jfile:
            if "simulated_" in file_output:
                continue
            if "simulated_" not in file_output:
                filtered_meta = meta_df[meta_df['index']==basename]
                try:
                    ground_truth_list = ast.literal_eval(list(filtered_meta['abundance'])[0])
                    ground_truth_list.extend(ast.literal_eval(list(filtered_meta['variant'])[0]))
                    ground_truth = ' '.join([str(x) for x in ground_truth_list])
                    ground_truth = standardize_ground_truth(ground_truth)
                except:
                    continue 
            if "simulated_" in file_output:
                ground_truth = basename.split("_")
                ground_truth = " ".join(ground_truth)
            #we haven't accounted for this filetype yet
            if ground_truth == 0 or 'aaron' in ground_truth:
                continue            
            data = json.load(jfile)
            keys = list(data.keys())
            print(ground_truth)
            #key is primer, value is percenets, mutations, haplotypes
            for k,v in data.items():
                #if dictionary is empty keep going
                if len(v) == 0:
                    continue 
                total_amp_depth = v['found_amps']
                 
                #related to the halotypes
                if total_amp_depth < 10:
                    continue    
                haplotype_percents = v['haplotype_percents']
                positions = v['poi']
                position_freq = v['poi_freq']
                mutations_first = v['mut_groups']
                
                #keep going if we have no mutations on this amplicon
                if len(haplotype_percents) == 0:
                    continue
                               
                mutations = mutations_first
                positions_to_remove = []

                #let's get rid of match haplotypes, low freq haplotypes
                for i, (hp, mg) in enumerate(zip(haplotype_percents, mutations)):
                    #figure out if this "haplotype" is somewhow all muts
                    uniq_muts = np.unique(mg)
                    if len(uniq_muts) == 1 and uniq_muts[0] == 'M':
                        positions_to_remove.append(i)
                        continue
                    if hp < 0.03 or hp > 0.97:
                        positions_to_remove.append(i)
               
                mutations = [a for i,a in enumerate(mutations) \
                        if i not in positions_to_remove]
                haplotype_percents = [a for i,a in enumerate(haplotype_percents) \
                        if i not in positions_to_remove]

                if len(haplotype_percents) == 0:
                    continue
               
                #try and see if this amplicon is messed up
                binding_correct = test_amplicon(k, haplotype_percents, position_freq, mutations, positions)
                done = False
                if binding_correct is False:
                    #print(k, position_freq, mutations, positions, haplotype_percents)                
                    for pk, pv in primer_dict.items():
                        if int(pv[0]) == int(k): 
                            improper_primers.append(pk)
                            if pk in mismatched_primers:
                                done = True
                if done is False:    
                    thf_condensed.append(haplotype_percents) 
                    mf_condensed.append(position_freq)
                    total_positions_extend.extend(positions)
                    total_haplotype_freq.extend(haplotype_percents)
                    total_amps.extend([k]*len(haplotype_percents))
                    total_mutation_freq.extend(position_freq)   
                    total_positions.append(positions)
      
            cluster_center_sums = []
            all_cluster_centers = []
            all_sil=[]
            all_labels = []
            all_bic = []
            
            total_haplotype_reshape = np.array(total_haplotype_freq).reshape(-1,1)
            total_mutation_flat = [item for sublist in total_mutation_freq for item in sublist]
            total_mutation_reshape = np.array([item for sublist in total_mutation_freq for item in sublist]).reshape(-1,1)

            lowest_value_highest_cluster = []
            highest_value_highest_cluster = [] 
            if len(total_haplotype_reshape) ==  0:
                continue
            possible_explanations = list(range(2,6)) 
            for num in possible_explanations:                          
                #kmeans clustering
                kmeans = KMeans(n_clusters=num, random_state=10).fit(total_haplotype_reshape)
                centers = kmeans.cluster_centers_            
                flat_list = [item for sublist in centers for item in sublist]
                all_cluster_centers.append(flat_list)                    
                all_labels.append(kmeans.labels_)                

                #here we find the smallest value in the "highest freq" cluster
                largest_cluster_center = max(flat_list)
                label_largest_cluster = flat_list.index(largest_cluster_center)
                smallest_value_largest_cluster = \
                    [v for v,l in zip(total_haplotype_freq, kmeans.labels_) if l == label_largest_cluster]
                 
                lowest_value_highest_cluster.append(min(smallest_value_largest_cluster))
                
                #this would be the "less ambiguous" method of calling consensus 
                highest_value_highest_cluster.append(max(smallest_value_largest_cluster))
                 
                cluster_center_sums.append(sum(flat_list)) 
                all_sil.append(silhouette_score(total_haplotype_reshape, kmeans.labels_))
                all_bic.append(compute_bic(kmeans, total_haplotype_reshape))
               
                    
            #now we have negative files in place
            if len(all_sil) != 0:        
                best_fit = max(all_sil)
                loc = all_sil.index(best_fit)

                #best_fit = min(all_bic)
                #loc = all_bic.index(best_fit)

                cluster_centers=all_cluster_centers[loc]
                cluster_opt = possible_explanations[loc]
                possible_threshold_low = lowest_value_highest_cluster[loc]
                possible_threshold_high = highest_value_highest_cluster[loc]
                act_sil = all_sil[loc]
                act_bic = all_bic[loc]
                used_labels = all_labels[loc]
                 
                mismatched_primer_starts = []
                for k, v in primer_dict.items():
                    if k in mismatched_primers:
                        mismatched_primer_starts.append(v[0])
                freq_flagged_starts = [] 
                for k, v in primer_dict.items():
                    if k in improper_primers:
                        freq_flagged_starts.append(v[0])
                 
                #let's graph the used labels and the haplotype freq/positons
                graph_clusters(used_labels, cluster_centers, thf_condensed, total_positions, mf_condensed,
                    total_haplotype_freq, total_positions_extend, basename, ground_truth, total_amps, \
                    mismatched_primer_starts, freq_flagged_starts)   

                #print(possible_threshold_high, possible_threshold_low)
                cluster_centers.sort(reverse=True)
                try:
                    possible_threshold_amb = possible_threshold_high+0.015
                    possible_threshold = possible_threshold_low-0.015
                except:
                    possible_threshold=0
                    possible_threshold_amb=0
                
                overlap_primers = [x for x in improper_primers if x in mismatched_primers]
                temp_dict[file_output.replace(".json","").replace(file_folder+'/',"")]= {
                     'cluster_centers': cluster_centers,\
                     'sil_opt_cluster':cluster_opt,\
                     'sil':act_sil, 'bic':act_bic, \
                     'threshold_low':possible_threshold, "threshold_high":possible_threshold_amb, \
                     'masked_primers': mismatched_primers, 'frequency_flagged_primers': improper_primers, \
                     'poor_primers': overlap_primers}
                            
            else: 
                temp_dict[file_output.replace(".json","").replace(file_folder+"/","")]= {
                    'cluster_centers': 0,\
                    'sil_opt_cluster':0,\
                    'sil':0,\
                    'threshold_low':0, 'threshold_high':0, 'masked_primers': 0, \
                    'frequency_flagged_primers': 0, 'poor_primers':0 }
        if TEST:
            print(temp_dict)
    df_outcome = pd.DataFrame(temp_dict).T
    df_outcome = df_outcome.reset_index()
    final_df = df_outcome.merge(meta_df, on='index', how='left')
    final_df.to_csv("snv_output.csv")
   
def test_amplicon(primer_0, haplotype_percents, position_freq, mutations, pos):
    """
    Parameters
    ----------
    primer_0 : int
    haplotype_percents : list
    positions_freq : list 
    mutations : list

    Returns
    -------
    correct : boolean
        Describes whether or not the primer binding is correct. 

    Compare the haplotype frequencies to the frequencies of the individual mutations
    that occur, and if they differ siginficantly return False.
    """
    correct=True
    #we iterate through each haplotype frequency
    for hp,mut,pf in zip(haplotype_percents, mutations, position_freq):
        for f in pf:
            #uninformative frequencies
            if f == 0 or f == -1 or f > 0.97:
                continue
            if abs(f-hp) > 0.15:
                correct=False
        if correct is False:
            #print(primer_0, haplotype_percents, position_freq, mutations,pos)
            return(correct)
    return(True)

def find_pure_file_mutations(pure_files, metadata, df):
    """
    Given a list of the pure filenames return mutations found.
    """
    seqNames = df['seqName'].tolist()
    seqNames_filt = ["file_"+x.split("_")[2] for x in seqNames]
    percent = [x.split("_")[3] for x in seqNames]
    df['index'] = seqNames_filt
    df['threshold'] = percent
    df_filtered = df[df['index'].isin(pure_files)]
    df_filtered = df_filtered[df_filtered['threshold']=='0.5'] 
    metadata_filtered = df_filtered.merge(metadata, on='index', how='left')
    
    gt_dict = {key:{'sub':[], 'dele':[], 'ins':[], 'fraction':{}, 'total':0} for key \
            in np.unique([ast.literal_eval(x)[0].lower() \
            for x in metadata_filtered['variant'].tolist()])}
    
    for index, row in metadata_filtered.iterrows():
        try:
            ins = row['aaInsertions'].split(',')
        except:
            ins=[]
        try:
            dele = row['aaDeletions'].split(',')
        except:
            dele=[]
        try:
            sub = row['aaSubstitutions'].split(',')
        except:
            sub=[]

        variant = ast.literal_eval(row['variant'])[0].lower()
        gt_dict[variant]['sub'].extend(sub)
        gt_dict[variant]['dele'].extend(dele)
        gt_dict[variant]['ins'].extend(ins)
        gt_dict[variant]['total'] += 1

    return(gt_dict)

def get_aa_from_position(positions):
    """
    Given a list of positions return the gene:aa code.
    """
    gene_table = "../gene_result.txt"
    gene_df = pd.read_table(gene_table, usecols=['start_position_on_the_genomic_accession', \
        'end_position_on_the_genomic_accession', 'Symbol'])    
   
    aa_list = []
    for position in positions: 
        for index, row in gene_df.iterrows():
            start = int(row['start_position_on_the_genomic_accession'])
            end = int(row['end_position_on_the_genomic_accession'])
            if start < position < end:
                gene_str = row['Symbol']
                aa = int((position-start)/3)+1
                aa_list.append(gene_str + ":" + str(aa))

    return(aa_list)

def parse_key_mutations():
    mutations_table = "../key_mutations.csv"
    gene_table = "../gene_result.txt"

    df = pd.read_csv(mutations_table)

    gene_df = pd.read_table(gene_table, usecols=['start_position_on_the_genomic_accession', \
        'end_position_on_the_genomic_accession', 'Symbol'])    

    total_pos_dict = {}
    for index, row in df.iterrows():
        gene = row['gene']
        aa = row['amino acid']
        
        if gene == 'ORF1b':
            gene = 'ORF1ab'

        if 'DEL' not in aa and 'del' not in aa:
            aa_loc = int(aa[1:-1])
    
        start=''
        for i, r in gene_df.iterrows():
            if gene in r['Symbol']:
                start = int(r['start_position_on_the_genomic_accession'])
                break
            
        lineage = row['lineage']
        total_pos = (aa_loc*3)+start-3 
        total_pos_list = [total_pos-1, total_pos, total_pos+1]
        
        if lineage not in total_pos_dict:
            total_pos_dict[lineage] = total_pos_list
        else:
            total_pos_dict[lineage].extend(total_pos_list)

    return(total_pos_dict)

def graph_clusters(used_labels, used_centers, thf_condensed, positions, mutations, \
    total_haplotype_freq, total_positions_extend, basename, ground_truth, amps, mismatched_primer_starts, \
    freq_flagged_starts):
    """
    Function takes in the labels, cluster labels, and positions and graphs them.
    """
    strain_1 = ground_truth.split(" ")[1]
    strain_2 = ground_truth.split(" ")[2]
     
    strain_1_muts = get_mutations_vcf(strain_1)
    if strain_2 != "none":
        strain_2_muts = get_mutations_vcf(strain_2)
        overlap = [x for x in strain_1_muts if x in strain_2_muts]
    else:
        overlap = []
    total_pos_dict = {}
    total_pos_dict[strain_1] = [x for x in strain_1_muts if x not in overlap]
    if strain_2 != "none":
        total_pos_dict[strain_2] = [x for x in strain_2_muts if x not in overlap]
    total_pos_dict['both linages'] = overlap

    #total_pos_dict = parse_key_mutations() 
    count = 0
        
    total_lineages = []
    pos_refactor = []
    """
    print("mutations", len(mutations), \
          "postions", len(positions), \
          "thf condensed", len(thf_condensed)
         )
    """
    
    #let's expand to associate certain lineages with a haplotype
    for c, (pos, muts, thf) in enumerate(zip(positions, mutations, thf_condensed)):
        #print(pos, 'muts', muts, 'thf', thf)
        
        #this iterates at the haplotype level
        for z,(mut,hf) in enumerate(zip(muts,thf)):
            remove = []
            for i,m in enumerate(mut):
                if m == 0:
                    remove.append(i)
            
            p = [x for i,x in enumerate(pos) if i not in remove]    
                     
            found_lin = "None" 
            for p1 in p:
                p1 += 1
                label = used_labels[count]
                for k,v in total_pos_dict.items():
                    if p1 in v:
                        found_lin = k
                        break
                if found_lin != "None":
                    break
            pos_refactor.append(p)
            total_lineages.append(found_lin)
    
    """    
    print(
        "total lineages", len(total_lineages), "\n",
        "used labels", len(used_labels), "\n",
        "total haplotype freq", len(total_haplotype_freq), "\n",
        "total positions", len(total_positions_extend), "\n",
        "position refactor", len(pos_refactor)
    )
    """
    unique_lineages = list(np.unique(total_lineages))
    unique_labels = list(np.unique(used_labels))
    combination_total = []
    for ul in unique_lineages:
        for lab in unique_labels:
            combination_total.append(str(ul)+"_"+str(lab)) 

    line_1 = [] #cluster larger strain 1
    line_2 = [] #cluster larger strain 2
    line_3 = [] #cluster smaller strain 1
    line_4 = [] #cluster smaller strain 2
    line_5 = [] #cluster smallest strain 1
    line_6 = [] #cluster smallest strain 2

    line_7 = [] #cluster smaller strain 1
    line_8 = [] #cluster smaller strain 2
    ordered_centers = copy.deepcopy(used_centers)
    ordered_centers.sort(reverse=True)
    
    if len(used_centers) == 2:
        cluster_1_label = used_centers.index(ordered_centers[0])
        cluster_2_label = used_centers.index(ordered_centers[1])
    elif len(used_centers) == 3:
        cluster_1_label = used_centers.index(ordered_centers[0])
        cluster_2_label = used_centers.index(ordered_centers[2])
        cluster_3_label = used_centers.index(ordered_centers[2])
    else:
        print(len(used_centers))
        return
    gt_list = ground_truth.split(" ")
    strain_1_label = gt_list[2] #expected larger strain
    strain_2_label = gt_list[1] #expected smaller strain
    df = pd.DataFrame({"charactertistic lineage":total_lineages, "cluster":used_labels, \
        "frequency":total_haplotype_freq, \
        'cluster - strain': [str(x) + '-' + str(y) for x,y in zip(used_labels, total_lineages)]})

    
    for tl, ul, thf in zip(total_lineages, used_labels, \
        total_haplotype_freq):
 
        if ul == cluster_1_label and tl == strain_1_label:
            line_1.append(thf)
        elif ul == cluster_2_label and tl == strain_1_label:
            line_2.append(thf)
        elif ul == cluster_1_label and tl == strain_2_label:
            line_3.append(thf)
        elif ul == cluster_2_label and tl ==strain_2_label:
            line_4.append(thf)
        elif ul == cluster_1_label and tl == "None":
            line_5.append(thf)
        elif ul == cluster_2_label and tl == "None":
            line_6.append(thf)
        elif ul == cluster_1_label and tl == "both lineages":
            line_7.append(thf)
        elif ul == cluster_2_label and tl == "both lineages":
            line_8.append(thf)


    plt.clf()
    plt.close()

    #plt.figure(figsize=(12, 12), dpi=80)
    fig, (ax1, ax3, ax5, ax7) = plt.subplots(4, 1, figsize=(8,8))
 
    #for strain 1
    sns.kdeplot(x=line_1, ax = ax1, color='purple', shade=True)
    sns.kdeplot(x=line_2, ax = ax1, color='orange', shade=True)
    plt.setp(ax1, xlim=(0,1))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set(xlabel=None)
    ax1.set(ylabel='%s Mutations' %strain_1_label)
    ax1.set(yticklabels=[])
    ax1.set(xticklabels=[])
    ax1.tick_params('both', length=0)

    #for strain 2
    sns.kdeplot(x=line_3, ax = ax3, color='purple', shade = True)
    sns.kdeplot(x=line_4, ax = ax3, color='orange', shade = True)
    plt.setp(ax3, xlim=(0,1))
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set(xlabel=None)
    ax3.set(ylabel='%s Mutations' %strain_2_label)
    ax3.set(yticklabels=[])
    ax3.set(xticklabels=[])   
    ax3.tick_params('both', length=0)
  
    #None labeled mutations
    sns.kdeplot(x=line_5, ax = ax5, color='purple', shade=True)
    sns.kdeplot(x=line_6, ax = ax5, color='orange', shade=True)
    plt.setp(ax5, xlim=(0,1))
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.set(xlabel=None)
    ax5.set(ylabel="Simulated Noise")
    ax5.set(yticklabels=[])   
    ax5.set(xticklabels=[])   
    ax5.tick_params('both', length=0)
    
    #both lineages labeled mutations
    sns.kdeplot(x=line_7, ax = ax7, color='purple', shade=True)
    sns.kdeplot(x=line_8, ax = ax7, color='orange', shade=True)
    plt.setp(ax7, xlim=(0,1))
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    ax7.set(xlabel="Haplotype Frequency")
    ax7.set(ylabel='Both Lineages')
    ax7.set(yticklabels=[])    
    ax7.tick_params('y', length=0)

 
    #sns.despine(top=True, right=True, left=True, bottom=False)
    #plt.title("%s Clustering Results by Lineage Haplotype and Frequency" %ground_truth, fontsize=7) 
    #plt.xlabel("Haplotype Frequency") 
    purple_patch = mpatches.Patch(color='purple', label='Cluster 1')
    orange_patch = mpatches.Patch(color='orange', label='Cluster 2')
    plt.savefig("./figures/%s_event_plot.png" %ground_truth)
    plt.clf()
    plt.close()
    
     
def get_mutations_vcf(strain):
    """
    Parameters
    ----------
    strain : str 
        The strain to parse.

    Returns
    -------
    mut_positions : list
        List of positions where we have mutations.

    Given a straing open the corresponding vcf file and return
    a list of expected mutation positions.    
    """
    vcf_file_location = "../simulated_data/simulated_vcf/%s_mod.vcf" %strain
    mut_positions = []
    with open(vcf_file_location, 'r') as vfile:
        for line in vfile:
            line = line.strip()
            if line.startswith("#"):
                continue
            line_list = line.split("\t")
            mut_positions.append(int(line_list[1]))
    return(mut_positions)

def standardize_ground_truth(label):
    """
    Parameters
    ----------
    label : str
    """
    order_list = ['50/50', '45/55', '40/60', \
              '35/65', '30/70', '25/75', '20/80', \
              '15/85', '10/90', '05/95', '0/100']
    strain_1 = ''
    strain_2 = ''
    percent_1 = ''
    percent_2 = ''
    label_list = label.split(" ")
 
    if "100.0" in label:
        label_list = label.split(" ")
        strain_1 = label_list[1].lower()
        strain_2 = "none"
        percent_1 = str(100)
        percent_2 = str(0)
        return_string  = "wastewater " + strain_1 + " " + strain_2 + " " + percent_1 + " " + percent_2
    elif len(label_list) == 4:
        percent_1 = str(int(float(label_list[0])))
        percent_2 = str(int(float(label_list[1])))
        strain_1 = label_list[2].lower()
        strain_2 = label_list[3].lower()
        return_string  = "wastewater " + strain_1 + " " + strain_2 + " " + percent_1 + " " + percent_2
    elif len(label_list) == 6:
        percent_1 = "33"
        percent_2 = "33"
        percent_3 = "33"
        strain_1 = label_list[3].lower()
        strain_2 = label_list[4].lower() 
        strain_3 = label_list[5].lower() 
        return_string  = "wastewater " + strain_1 + " " + strain_2 + " " + strain_3 + " " + percent_1 + " " + percent_2 + " " + percent_3
    elif len(label_list) == 8:
        percent_1 = "25"
        percent_2 = "25"
        percent_3 = "25"
        percent_4 = "25"
        strain_1 = label_list[4].lower()
        strain_2 = label_list[5].lower() 
        strain_3 = label_list[6].lower() 
        strain_4 = label_list[7].lower() 
        return_string  = "wastewater " + strain_1 + " " + strain_2 + " " + strain_3 + " " + strain_4 + " " + percent_1 + " " + percent_2 + " " + percent_3 + " " + percent_4
    elif len(label_list) == 10:
        percent_1 = "20"
        percent_2 = "20"
        percent_3 = "20"
        percent_4 = "20"
        percent_5 = "20"
        strain_1 = label_list[5].lower()
        strain_2 = label_list[6].lower() 
        strain_3 = label_list[7].lower() 
        strain_4 = label_list[8].lower() 
        strain_5 = label_list[9].lower() 
        return_string  = "wastewater " + strain_1 + " " + strain_2 + " " + strain_3 + " " + strain_4 + " " + strain_5 + " " + percent_1 + " " + percent_2 + " " + percent_3 + " " + percent_4 + " "+ percent_5
    else:
        return(0)
    
    return(return_string)

def reformat_metadata(df):
    """
    """
    
    columns = {"filename":[], "strain_1": [], "strain_2":[], "strain_3":[], "strain_4": [], \
        "strain_5":[], "percent_1":[], "percent_2":[], "percent_3":[], "percent_4":[], "percent_5":[], \
        "reads_1":[], "reads_2":[], "reads_3":[], "reads_4":[], "reads_5":[], "total_reads":[]} 
    for index, row in df.iterrows():
        try:
            variants = ast.literal_eval(row['variant'])
        except:
            continue
        abundances = ast.literal_eval(row['abundance(%)'])
        i = 0
        temp_abundances = ["None"]*5
        temp_variants = ["None"]*5
        temp_abundances[0] = 0
        temp_abundances[1] = 0
        for v, a in zip(variants, abundances):
            temp_abundances[i] = a
            temp_variants[i] = v.lower()
            i += 1 
        columns['filename'].append(row['filename'].replace(".bam",""))
        columns['percent_1'].append(temp_abundances[0])
        columns['percent_2'].append(temp_abundances[1])
        columns['percent_3'].append(temp_abundances[2])
        columns['percent_4'].append(temp_abundances[3])
        columns['percent_5'].append(temp_abundances[4])
        columns['strain_1'].append(temp_variants[0])
        columns['strain_2'].append(temp_variants[1])
        columns['strain_3'].append(temp_variants[2])
        columns['strain_4'].append(temp_variants[3])
        columns['strain_5'].append(temp_variants[4])
        columns['reads_1'].append(temp_abundances[0])
        columns['reads_2'].append(temp_abundances[1])
        columns['reads_3'].append(temp_abundances[2])
        columns['reads_4'].append(temp_abundances[3])
        columns['reads_5'].append(temp_abundances[4])
        columns['total_reads'].append(100)
    df = pd.DataFrame(columns)
    return(df)

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

if __name__ == "__main__":
    main()
