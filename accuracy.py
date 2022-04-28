import os
import sys
import ast
import json
import numpy as np
import pandas as pd
import seaborn as sns
import Levenshtein as lv
from contaminant_analysis import calculate_positional_depths
from process_spike_in import consensus_call

def find_ground_truth(filename, peak, \
    spreadsheet="./spike_in/spike-in_bams_spikein_metadata.csv"):
    """
    Given a file and a peak, find the name of what it's supposed to be.
    """
    
    bam_file_match = filename + ".bam"
    gt_df = pd.read_csv(spreadsheet)
    for index, row in gt_df.iterrows():
        if row['filename'] == bam_file_match:
            gt_var = row['variant']
            gt_abundance = row['abundance(%)']
            gt_var = ast.literal_eval(gt_var)
            gt_abundance = [float(item)/100 for item in ast.literal_eval(gt_abundance)]
            closest_peak_index = min(range(len(gt_abundance)), \
                key=lambda i: abs(gt_abundance[i]-float(peak)))
            closest_peak = gt_abundance[closest_peak_index]
            
            closest_strain = gt_var[closest_peak_index]
            break
            
            
    return(closest_strain, closest_peak)


def calculate_accuracy(gt_strain, peak_consensus_file, predicted_strain):
    """
    hard-code things to start, then 
    """
    gt_strain = gt_strain.lower()
   
    delta_pure_file = "./spike_in/bam/file_324.calmd.bam"
    #try and see if we've calcualted consensus already, if not calculate it
    if not os.path.isfile("./ground_truth_consensus/delta_pure.fasta"):
        print("Ground truth consensus not found, creating file")
        sys.exit(0)
        delta_pos_depths = calculate_positional_depths(delta_pure_file)
        delta_consensus_string = consensus_call({}, delta_pos_depths)
        with open("./ground_truth_consensus/delta_pure.fasta","w") as ffile:
            #write header
            ffile.write(">delta_pure\n")
            ffile.write(delta_consensus_string)
    #we already have the consensus, open it!
    else:
        with open("./ground_truth_consensus/%s_pure.fasta" %gt_strain, "r") as ffile:
            for count, line in enumerate(ffile):
                #header line
                if count == 0:
                    continue
                delta_consensus_string = line.strip()
    
    #load the consensus string from the file
    with open(peak_consensus_file, "r") as ffile:
        for count,line in enumerate(ffile):        
            #header line
            if count == 0:
                continue
            peak_consensus_string = line.strip()
    
    #calculate distance between them, clip to proper length since they're aligned else
    """
    if len(peak_consensus_string) != len(delta_consensus_string):
        print(len(peak_consensus_string), len(delta_consensus_string), gt_strain, predicted_strain)
        print("Error, not the same length!")
        if len(peak_consensus_string) > len(delta_consensus_string):
            peak_consensus_string = peak_consensus_string[:len(delta_consensus_string)]
        else:
            delta_consensus_string = delta_consensus_string[:len(peak_consensus_string)]
    """ 
    #smaller is better!
    lev_distance = 1 - lv.ratio(peak_consensus_string, delta_consensus_string)
    return(lev_distance)
    
def main():
    #get all result dirs
    all_files = [item for item in os.listdir("./") if os.path.isdir(item) and "file" in item]
    
    final_df_dict = {"filename":[], "peak_value":[], "clade":[], "nextclade_pango":[], \
        "gt_value":[], "gt":[], "peak_width":[], "lev_dist":[]}
    for directory in all_files:
        print("calc accuracy for ", directory)
        nextclade_loc = os.path.join(directory, "nextclade.tsv")
        add_info_json = os.path.join(directory, "peak_information.json")
        
        nextclade_df = pd.read_table(nextclade_loc)
        with open(add_info_json, 'r') as jfile:
            data = json.load(jfile)

        #print(nextclade_df.columns)
        for index, row in nextclade_df.iterrows():
            lineage  = row['Nextclade_pango']
            seq_name = row['seqName']
            clade = row['clade']
            peak_value = seq_name.split("_")[-2]
            gt, gt_value = find_ground_truth(directory, peak_value)            
            
            peak_width = data['width-heights'][data['peak-value'].index(float(peak_value))]
                                 
            final_df_dict['filename'].append(directory)
            final_df_dict['peak_value'].append(peak_value)
            final_df_dict['clade'].append(clade)
            final_df_dict['nextclade_pango'].append(lineage)
            final_df_dict['gt_value'].append(gt_value)
            final_df_dict['gt'].append(gt)
            final_df_dict['peak_width'].append(round(peak_width, 2))   
            
            cf_string = "consensus_" + directory + "_" + str(peak_value) + ".fasta"
            consensus_filename = os.path.join(directory, cf_string)
            
            #align the consensus fasta to the 'ref'
            gt_con_file = "ground_truth_consensus/%s_pure.fasta" %(gt.lower())
            cmd = "bwa mem %s %s > aln-se.sam" %(gt_con_file, consensus_filename)
            os.system(cmd)
            cmd2 = "samtools fasta aln-se.sam > output.fasta"            
            os.system(cmd2)
            
            print("GT Strain: ", gt, " ", gt_value)
            print("Predicted Strain: ", clade, " ", peak_value)
        
            lev_dist = calculate_accuracy(gt, "output.fasta", clade)        
            final_df_dict["lev_dist"].append(lev_dist)
            
            
        

    df_out = pd.DataFrame(final_df_dict)
    print(df_out)
    df_out.to_csv("out_csv.csv")
        
        
if __name__ == "__main__":
    main()
