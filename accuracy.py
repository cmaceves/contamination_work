import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import Levenshtein as lv
from contaminant_analysis import calculate_positional_depths
from process_spike_in import consensus_call

def find_ground_truth(filename, peak, peak_prediction, \
    spreadsheet="./spike_in/spike-in_bams_spikein_metadata.csv"):
    """
    Given a file and a peak, find the name of what it's supposed to be.
    """
    bam_file_match = filename + ".bam"
    gt_df = pd.read_csv(spreadsheet)
    for index, row in gt_df.iterrows():
        if row['filename'] == bam_file_match:
            print(row)
            sys.exit(0)



def calculate_accuracy():
    """
    hard-code things to start, then 
    """
    #looking at delta peak
    delta_pure_file = "./spike_in/bam/file_324.calmd.bam"
    
    #try and see if we've calcualted consensus already, if not calculate it
    if not os.path.isfile("./ground_truth_consensus/delta_pure.fasta"):
        print("Ground truth consensus not found, creating file")
        delta_pos_depths = calculate_positional_depths(delta_pure_file)
        delta_consensus_string = consensus_call({}, delta_pos_depths)
        with open("./ground_truth_consensus/delta_pure.fasta","w") as ffile:
            #write header
            ffile.write(">delta_pure\n")
            ffile.write(delta_consensus_string)
    #we already have the consensus, open it!
    else:
        print("Ground truth consensus found, opening file.")
        with open("./ground_truth_consensus/delta_pure.fasta", "r") as ffile:
            for count, line in enumerate(ffile):
                #header line
                if count == 0:
                    continue
                delta_consensus_string = line.strip()
             
    #load the consensus string from the file
    peak_consensus_file = "consensus_file_256/consensus_file_256_0.475.fasta"
    with open(peak_consensus_file, "r") as ffile:
        for count,line in enumerate(ffile):        
            #header line
            if count == 0:
                continue
            peak_consensus_string = line.strip()
    
    #calculate distance between them, exit if not same length
    if len(peak_consensus_string) != len(delta_consensus_string):
        print("Error, not the same length!")
        sys.exit(1)

    #smaller is better!
    lev_distance = 1 - lv.ratio(peak_consensus_string, delta_consensus_string)
    print(lev_distance)

def main():
    filename = "file_256"
    peak_value = "0.475"
    peak_prediction = "delta"

    

    #find_ground_truth(filename, peak_value, peak_prediction)
    sys.exit(0)
    calculate_accuracy()

if __name__ == "__main__":
    main()
