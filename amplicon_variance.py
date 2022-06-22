import os
import sys
import ast
import copy
import json
import pysam
import warnings
import argparse
import statistics
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from util import calculate_positional_depths

def warn(*args, **kwargs):
    pass

warnings.warn = warn

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
    #adjust for ends of possible reads
    primer_0 -= 1
    primer_1 += 1 
   
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

                total_ref = pileupread.alignment.get_reference_positions(full_length=True)
                total_query = list(pileupread.alignment.query_sequence)
                total_qualities = list(pileupread.alignment.query_qualities)

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

                #used to track blocks of deletions and insertions
                store_nuc=''
                on_insertion=False
                on_deletion=False

                for pcount,(pos,nuc,qual,cigtype) in enumerate(zip(total_ref, total_query, total_qualities, expand_cigar)):
                    if pos not in amp_indexes:
                        continue

                    #if we have an insertion we create a seperate encoding for it, and place it in
                    nuc = nuc.upper()
                    
                    if cigtype == 1:
                        on_insertion = True
                        store_nuc += nuc.upper()
                        continue
                    if cigtype == 4 or cigtype == 5:
                        continue

                    if cigtype == 2:

                    #we've hit the end of the insertion
                    elif cigtype == 0 and on_insertion is True:
                        new_value = max(encoded_nucs, key=encoded_nucs.get) 
                        nuc = '+'+store_nuc
                        encoded_nucs[nuc] = encoded_nucs[new_value]+1
                        on_insertion = False
                        pos = pcount - len(store_nuc)
                        store_nuc =''
   
                    if pos in amp_indexes:
                        if int(pos) in noise_dict:
                            noise = noise_dict[int(pos)]
                            if nuc in noise:                           
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
                            if temp > 0.97:
                                loc = amp_indexes.index(pos)
                                amp[loc] = 0.0
                                freq[loc] =0.0
                                quality[loc]=0.0
                                continue
                            else:
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

    #estimated num groups 
    num_groups = []
    final_tuples = []
    poi = []
    #position of interest within this matrix
    relative_poi=[]

    #iterate by position
    for c, (pos_array, qual_array) in enumerate(zip(read_matrix.T, quality_matrix.T)):
        gt_pos = c+primer_0
        if gt_pos != 23603:
            continue
        #positions that match the reference
        match_ref = pos_array[pos_array==0]
        #positions that are soft clipped
        soft_clipped = pos_array[pos_array==-1]
        #positions that have mutations
        mutations = pos_array[pos_array>0]


        #if we don't have mutations here we don't care
        if len(mutations) == 0:
            continue
        print("soft clipped:", len(soft_clipped))
        print("match ref:", len(match_ref))
        print("mutations:", len(mutations))

        
        #remover blank values from quality
        filter_qual = qual_array[qual_array>0]
        #find the average quality here
        avg_qual = np.average(filter_qual)
        if avg_qual < 20:
            continue        
        #depth doesn't include the softclipped
        total_depth = len(match_ref) + len(mutations)

        values, counts = np.unique(mutations, return_counts=True)
        percent = [x/total_depth for x in counts]
        #remove low level mutations
        final = [z for z in zip(percent, values) if z[0] > 0.03]
        if len(final) > 0:
            relative_poi.append(c)
            poi.append(c+primer_0)
        num_groups.append(len(final))
        final_tuples.append(final)
 
    if read_matrix.ndim < 2:
        read_matrix = np.zeros((0,0))  
   
    max_groups = []
    group_counts=[]
    if len(relative_poi) > 0:
        filter_matrix = freq_matrix.T[relative_poi]
        read_matrix_filt = read_matrix.T[relative_poi]
        quality_matrix_filt = quality_matrix.T[relative_poi]

        if filter_matrix.shape != read_matrix_filt.shape != quality_matrix_filt.shape:
            print("error in matrix shape.")
            sys.exit(1)

        #string rep
        groups =[]
        #float rep
        groups2=[]
        #total used
        cc = 0
        for count, (thing) in enumerate(filter_matrix.T):
            #not covered or soft clipped, skip it, and later we'll place it in nearest group
            if -1 in thing:
                continue
            cc += 1
            thing = [str(x) for x in list(thing)]
            stringify = '-'.join(thing)
            thing = [float(x) for x in thing]
            if stringify not in groups:
                groups.append(stringify)
                groups2.append(list(thing))
                group_counts.append(1)
            else:
                loc = groups.index(stringify)
                group_counts[int(loc)] += 1
                 
    group_percents = [x/cc for x in group_counts]     
    
    for i,(mg, gc) in enumerate(zip(max_groups, group_percents)):
        if np.count_nonzero(mg) == 0:
            max_groups.remove(mg)
            group_percents.remove(gc)

    return(poi, group_percents, found_amps)


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
    """
    del primer_pairs['covid19genome_0-200_s0_M1']
    del primer_pairs['covid19genome_0-200_s0_M2']
    df = pd.DataFrame(primer_pairs)
    df.T.to_csv("primer_pairs.tsv", sep='\t', header=False, index=False)
    """

    return(primer_dict, primer_dict_inner)


def parallel(all_bams):
    results = Parallel(n_jobs=3)(delayed(process)(bam) for bam in all_bams)
    
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
        bam_filename = "../spike_in/" + filename + "_sorted.calmd.bam"
        file_threshold_dict[bam_filename] = {"threshold": round(thresh,2), "output": "../consensus/"+\
            filename +"_"+ str(round(thresh,2)), 'other_thresh':round(other_thresh,2), "other_output": \
            "../consensus/"+filename+"_"+str(round(other_thresh,2))}

    return(file_threshold_dict)

def main():
    #list all json files
    #all_json = [x for x in os.listdir("../spike_in") if x.endswith('.bam')]
    #all_json = ["file_124_sorted.calmd.bam", "file_125_sorted.calmd.bam", "file_127_sorted.calmd.bam"]
    #all_json = ["file_124_sorted.calmd.bam"]
    #all_json = [os.path.join("../spike_in",x) for x in all_json]
    
    file_folder = "../json"

    #parallel(all_json)
    #sys.exit(0)

    #this line is used for testing
    process("/home/chrissy/Desktop/retrimmed_bam/file_124.sorted.final.bam")
    sys.exit(0)
    
    #creates the .csv file with thresholds and other info 
    group_files_analysis(file_folder)
    sys.exit(0)

    #sys.exit(0)
    #parses the tsv containing thresholding information
    file_threshold_dict = parse_snv_output("snv_output.csv")
    """
    #remove old specific consensus thresholds
    con = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    for filename in os.listdir("../consensus"):
        filename_check = filename.split("_")[-1].replace(".fa","")
        if float(filename_check) not in con:
            os.system("rm %s" %(os.path.join("../consensus",filename)))
    """
    
    #calls consensus using the above parsed thresholds
    for k, v in file_threshold_dict.items():
        print(k, " opt thresh")
        threshold = v['threshold']
        output_filename = v['output']
        other_thresh = v['other_thresh']
        other_output=v['other_output']
         
        #if os.path.isfile(output_filename + ".fa"):
        #    continue
        if float(threshold) > 0.50:
            call_consensus(k, output_filename, threshold)            
        #if os.path.isfile(other_output + ".fa"):
        #    continue
        if float(other_thresh) > 0.5:
            call_consensus(k, other_output, other_thresh)            
    
    sys.exit(0) 
    #calls consensus using an array of thresholds 0.5-0.95 every 0.05 
    """
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
    #metadata_df = pd.read_csv("snv_output.csv")
    #analyze_nextstrain("nextclade.tsv", metadata_df)

def analyze_nextstrain(filename, metadata):
    """
    Parse apart nextclade file and analyze the consensus calling data.
    """

    #hard coded s gene mutations per strain
    s_gene_dict = {"alpha":[69,70,144,501,570,614,681,716,982,1118], \
        "beta":[80,215,241,243,417,484,501,614,701], \
        "delta":[19,156,157,158,452,478,614,681,950], \
        "gamma":[18,20,26,138,190,417,484,501,614,655,1027,1176]}
   
    #list of 100% files that got properly processed
    pure_files = ["file_320", "file_321", "file_323", "file_328", "file_331", "file_334", "file_337", \
            "file_338", "file_339"]

   
    df = pd.read_table(filename)
    test_file = "file_110"
     
    #find and return a list of expected mutations based on the pure files
    gt_dict = find_pure_file_mutations(pure_files, metadata, df)


    #plot where we chose to call consensus
    sil_scores = metadata['sil']
    filenames_meta = metadata['index'].tolist()
    meta_threshold = metadata['threshold'].tolist()
    amb_threshold = metadata['threshold_amb'].tolist() 
    sil_dict = {filenames_meta[i]: {'sil':sil_scores[i], 'thresh':round(meta_threshold[i],2), \
        'thresh_amb':amb_threshold[i]} for i in range(len(filenames_meta))}
    
    #plots varation in mutations found in technical replicates, and predicted thresholds
    #variation_in_technical_replicates(metadata, df)
    single_file_mutation_graph(["file_124","file_125","file_127"], gt_dict, metadata, df)

    #analyze primer binding sites
    file_test_list = ["file_124","file_125","file_127"]

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
    
def process(bam):
    """
    Parameters
    ----------
    bam : str
        Path to the bam of interest.
    """
    basename = bam.split("/")[-1].split(".")[0].replace("_sorted","")    
    primer_file = "../sarscov2_v2_primers.bed"  
    reference_filepath = "../sequence.fasta"
    variants_output_dir = "../variants"
    bed_filepath = "../sarscov2_v2_primers.bed"
    primer_pair_filepath = "../primer_pairs.tsv"
    masked_output_dir = "../masked"
    primer_drop_dir = "../primer_drops"
    
    """
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
    """
    primer_dict, primer_dict_inner  = get_primers(primer_file)
      
    if not os.path.isfile("../pos_depths/%s_pos_depths.json" %basename): 
        try:
            total_pos_depths = calculate_positional_depths(bam)
            with open("../pos_depths/%s_pos_depths.json" %basename, "w") as jfile:
                json.dump(total_pos_depths, jfile) 
        except:
            return(1)
    else:
        with open("../pos_depths/%s_pos_depths.json" %basename, "r") as jfile:
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
     
    #gives us matches of primers by name
    primer_dict, primer_dict_inner  = get_primers(primer_file)
 
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
     
        #TEST LINE
        if primer_0 != 23514:
            continue
 
        #we didn't successfully find a primer pair
        if primer_0 == 0.0 or primer_1 == 0.0 or primer_0_inner == 0 or primer_1_inner ==0:
            continue    
        p_0.append(primer_0)
        p_1.append(primer_1)
        p_0_inner.append(primer_0_inner)
        p_1_inner.append(primer_1_inner)
        poi, groups_percents, found_amps = extract_amplicons(bam, primer_0, primer_1, primer_0_inner, primer_1_inner, noise_dict, total_pos_depths)

        amplicon_level_dict = { "poi":poi, \
            "haplotype_percents": groups_percents, "found_amps": int(found_amps), \
            }
        file_level_dict[primer_0] = amplicon_level_dict
    
    with open("../json_primers/primers_%s.json" %basename, "w") as jfile:
        json.dump(file_level_dict, jfile)
    return(0)

def group_files_analysis(file_folder):  
    """
    file_folder : str
        Location where the .json files containing amplicon level mutations are found.
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

    all_files = [x for x in os.listdir("../spike_in") if x.endswith('sorted.calmd.bam')]
    json_filenames = [os.path.join(file_folder,(x.split('.')[0]+'.json').replace("_sorted","")) for x in all_files]
    
    #used the control what files we look at
    seen = [os.path.join(file_folder,"file_124.json")]
    #seen=[]
    temp_dict = {}
   
    variant_read_covariance = []
    all_mng_data = []

    #iterate through every output result
    for file_output in json_filenames:
        
        if not os.path.isfile(file_output):
            continue
        if file_output not in seen:
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
            d_data = []
            freq_groups = []
            collapsed_read_max = 0

            #key is primer, value is percenets, mutations, haplotypes
            for k,v in data.items():
                 
                if int(k) != 23514:
                    continue
                print(file_folder, v)
                sys.exit(0)
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
            
                #print(file_output, percents, mng)
                d_data.extend([i for i in percents if i > 0.03])
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
            
            #print(file_output)
            unique_freq.sort()             
            zipped = zip(x,y,mut_tag)
            zipped = sorted(zipped, key = lambda x:x[1])
            all_mng_data.append(zipped)
            try:
                x,y,mut_tag = zip(*zipped)
            except:
                print("fail ", file_output)
                #os.system("rm %s" %file_output)
                continue
            c_data = []
            all_muts_count = sum(y)
            for x1,y1,m1, in zip(x,y,mut_tag):
                c_data.extend([x1])
                #print(x1,y1,m1)
                pass
            
            #print(file_output, " c ", c_data)
            #print(file_output, " d ", d_data)
            #returns scores/targets 
            #combo_move = number_gen(possible_range, list(x), list(mut_tag))
            
            cluster_center_sums = []
            all_cluster_centers = []
            all_inertia = []
            cluster_sums=[]
            cluster_centers=[]
            all_sil=[]
            x_reshape = np.array(c_data).reshape(-1,1)
            d_reshape = np.array(d_data).reshape(-1,1)
            lowest_value_highest_cluster = []
            highest_value_highest_cluster = [] 
            #print(d_data, c_data)
            if len(x_reshape) == 1:
                possible_explanations = [1]
            elif len(x_reshape) <= 6:
                possible_explanations = list(range(max(halotypes),len(x_reshape)))       
            else:
                possible_explanations = list(range(max(halotypes),6)) 
             
            #print(possible_explanations, x, mut_tag) 
            for num in possible_explanations: 
                combo_move = number_gen([num], list(x), list(mut_tag))
                initial = np.array([float(x) for x in combo_move[1][0]]).reshape(-1,1)
                #kmeans clustering
                init_cluster = np.array(x[-num:]).reshape(-1,1)
                kmeans = KMeans(n_clusters=num, init=initial, random_state=10).fit(d_reshape)
                centers = kmeans.cluster_centers_            
                flat_list = [item for sublist in centers for item in sublist]
                
                all_cluster_centers.append(flat_list)                    
              
                #here we find the smallest value in the "highest freq" cluster
                largest_cluster_center = max(flat_list)
                label_largest_cluster = flat_list.index(largest_cluster_center)
                smallest_value_largest_cluster = \
                    [v for v,l in zip(d_data, kmeans.labels_) if l == label_largest_cluster]
                
                lowest_value_highest_cluster.append(min(smallest_value_largest_cluster))
                
               
                #this would be the "less ambiguous" method of calling consensus 
                highest_value_highest_cluster.append(max(smallest_value_largest_cluster))
                 
                cluster_center_sums.append(sum(flat_list))
                try: 
                    all_sil.append(silhouette_score(d_reshape, kmeans.labels_))
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
                """
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
                """
                max_halotype = max(halotypes)
                
                temp_dict[file_output.replace(".json","").replace(file_folder+'/',"")]= {
                    'max_halpotypes':max_halotype,\
                     'cluster_centers': cluster_centers,\
                     'sil_opt_cluster':cluster_opt,\
                     'sil':act_sil,\
                     'unique_mut':str(zipped), 'mut_certainty':mut_certainty,\
                     'threshold':possible_threshold, "threshold_amb":possible_threshold_amb}
                
            else: 
                temp_dict[file_output.replace(".json","").replace(file_folder+"/","")]= {
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

def variation_in_technical_replicates(metadata, df):
    """
    Creates plots looking at the variance of technical replicates.
    """
    technical_replicates = {}

    #generate dict of technical replicates
    for index, row in metadata.iterrows():
        filename = row['index']
        try:
            var = ast.literal_eval(row['variant'])
        except:
            var = ['Negative']
        abun = ast.literal_eval(row['abundance'])
        abun = [str(round(x,2)) for x in abun]
        var = [x.lower() for x in var]
        string_key = '_'.join(var) + '_' + '_'.join(abun)     
        if string_key in technical_replicates:
            technical_replicates[string_key].append(filename)
        else:
            technical_replicates[string_key] = [filename]
    
    #shows sils versus percent breakdowns
    sil_percent_dict = {}
    
    #for sil boxplots 
    box_x=[]
    box_y=[]

    #compare values for technical replicates
    for key, value in technical_replicates.items():
        var = key.split("_")
        var = var[:int(len(var)/2)]
        var = '_'.join(var)
        
        df_filtered = df[df['index'].isin(value)]
        meta_filtered = metadata[metadata['index'].isin(value)]
        #print(df_filtered)
        #print(meta_filtered.columns)
        #print(meta_filtered['threshold'])
        #print(meta_filtered['threshold_amb'])
        sils = meta_filtered['sil'].tolist()
        if len(sils) < 2:
            continue
   
        percent_breakdown = key.split("_")
        perc = []
        for pb in percent_breakdown:
            try:
                perc.append(float(pb))
            except:
                pass
        temp_key = ''
        if 95 in perc:
            temp_key='95/5'
        elif 100 in perc:
            temp_key='100'
        elif 90 in perc:
            temp_key='90/10'
        elif 80 in perc:
            temp_key='80/20'
        elif 60 in perc:
            temp_key='60/40'
        elif 50 in perc:
            temp_key='50/50'
        elif 33.33 in perc:
            temp_key='33*3'
        elif 25 in perc:
            temp_key='25*4'
        elif 20 in perc:
            temp_key='20*5'
        box_x.extend([temp_key]*len(sils))
        box_y.extend(sils)
        if temp_key in sil_percent_dict:
            if var in sil_percent_dict[temp_key]:
                sil_percent_dict[temp_key][var]=sils
            else:
                sil_percent_dict[temp_key][var] = sils
        else:
            sil_percent_dict[temp_key] = {var:sils}
    
    #relationship between percents and sil scores
    order= ['95/5','90/10','80/20','60/40','50/50','33*3','25*4','20*5']

    box_x_f = []
    box_y_f = []
    #we don't want to even visualize the 100% samples
    for i,(x,y) in enumerate(zip(box_x, box_y)):
        if str(x) == "100":
            continue
        else:
            box_x_f.append(x)
            box_y_f.append(y)

    sns.set_style("whitegrid")
    sns.boxplot(x=box_x_f, y=box_y_f, order=order)
    plt.xlabel("Ratio of Variants in Samples")
    plt.ylim(0,1)
    plt.ylabel("Silhouette Score")
    plt.axhline(y=0.80, color='orange',linewidth=2)
    #plt.title("relationship between sil score and variants ratio")
    plt.tight_layout() 
    plt.savefig("sil_ratio_boxplot.png")
    plt.close()
    
    all_var = []
    #variance among technical replicate sil scores
    for key, value in sil_percent_dict.items():  
        
        for k,sils in value.items():
            
            if len(sils) < 3:
                continue
            variance = statistics.variance(sils)
            if max(sils)-min(sils) > 0.1:
                print(k, key, sils)
            all_var.append(max(sils)-min(sils))
            all_var.append(variance)
    #print(all_var)
    sns.distplot(all_var)
    plt.xlabel("distance between max/min thresholds selected for the same file type (percent/variant)")
    plt.title("threshold similarity among the same file type")
    plt.savefig("dist_var.png")
    plt.close()

    sys.exit(0)

def single_file_mutation_graph(files, gt_dict, metadata, df):
    single_file = [files[0]]
    #files =['file_109', 'file_110']
    seqNames = df['seqName'].tolist()
    seqNames_filt = ["file_"+x.split("_")[2] for x in seqNames]
    percent = [x.split("_")[3] for x in seqNames]
    df['index'] = seqNames_filt
    df['threshold'] = percent
    df_filtered = df[df['index'].isin(files)]
    metadata_filtered = df_filtered.merge(metadata, on='index', how='left')
    gt = gt_dict['alpha']

    #handle the other element of the file
    gt_2 = gt_dict['gamma']
    val_sub_2 = [x for x in list(np.unique(gt_2['sub'])) if x.startswith("S")]
    val_ins_2 = [x for x in list(np.unique(gt_2['ins'])) if x.startswith("S")]
    val_del_2 = [x for x in list(np.unique(gt_2['dele'])) if x.startswith("S")]
    total_2=val_sub_2
    total_2.extend(val_ins_2)
    total_2.extend(val_del_2)

    val_sub = [x for x in list(np.unique(gt['sub'])) if x.startswith("S")]
    val_ins = [x for x in list(np.unique(gt['ins'])) if x.startswith("S")]
    val_del = [x for x in list(np.unique(gt['dele'])) if x.startswith("S")]
    total = val_sub
    total.extend(val_ins)
    total.extend(val_del)
    order = np.unique(metadata_filtered['threshold_x']).tolist()
    order.sort(reverse=True)
    
    shape = int(len(metadata_filtered)/len(files))
    heat = np.zeros((len(order),len(total)))
    heat_2 = np.zeros((len(order),len(total_2)))
    divisor = [0.0]*len(order)
    for index, row in metadata_filtered.iterrows():    
        try:
            ins = row['aaInsertions'].split(',')
        except:
            ins=[]
        try:
            #print(row['threshold_x'], row['aaDeletions'])
            dele = row['aaDeletions'].split(',')
        except:
            dele=[]
        try:
            #print(row['threshold_x'], row['aaSubstitutions'])
            sub = row['aaSubstitutions'].split(',')

        except:
            sub=[]
        loc2 =order.index(row['threshold_x'])
        divisor[loc2] += 1
        for i in ins:
            if i in val_ins_2:
                loc = total_2.index(i)
                heat_2[loc2][loc] += 1
            
            if i in val_ins:
                loc = total.index(i)
                heat[loc2][loc] += 1
        
        for s in sub:
            #check if the sub is in ground truth
            if s in val_sub:
                loc = total.index(s)
                heat[loc2][loc] += 1
            if s in val_sub_2:
                loc = total_2.index(s)
                heat_2[loc2][loc] += 1
        
        for d in dele:
            #check if the del is in ground truth
            if d in val_del:
                loc = total.index(d)
                loc2 =order.index(row['threshold_x'])
                heat[loc2][loc] += 1
            if d in val_del_2:
                loc = total_2.index(d)
                heat_2[loc2][loc] += 1
 
    heat = heat/np.array(divisor)[:,None]
    heat_2 = heat_2/np.array(divisor)[:,None]
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    sns.heatmap(heat, yticklabels=order, xticklabels=val_sub, square=False, ax=ax1, cbar=False)
    #ax1.axhline(y=0.98, color='orange',linewidth=2 )
    ax1.set(xlabel='alpha', ylabel='consensus threshold used')
    sns.heatmap(heat_2, yticklabels=order, xticklabels=val_sub_2, square=False, ax=ax2, cbar=False)
    ax2.set(xlabel='gamma')
     
    #plt.ylim(0,1)
    #plt.ylabel("consensus threshold used")
    #plt.axhline(y=0.80, color='orange',linewidth=2)
    #plt.title("10/90 mutations recovered gamma/alpha")
    plt.tight_layout() 
    plt.savefig("heat_10_90_gamma_alpha.png")
    plt.close()
      
if __name__ == "__main__":
    main()
