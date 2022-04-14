import os
import sys
import ast
import json
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import signal

#other script import
from contaminant_analysis import calculate_positional_depths, \
    calculate_read_probability, actually_call_pos_depths

def process_bam_file(bam, create_new_hdf5, name, threshold):
    print("working on %s" %bam)

    position_depths = calculate_positional_depths(bam)
    #with open('pos_depths_%s.json' %name ,'w') as jfile:
    #    json.dump(position_depths, jfile)
    
    #with open('pos_depths_%s.json' %name ,'r') as jfile:
    #    position_depths = json.load(jfile)
    
    calculate_read_probability(position_depths, bam,\
        name, threshold, create_new_hdf5, virus_1=None, virus_2=None, gt_dict=None)
    os.system("rm myfile_%s.hdf5" %name)
 
def main():
    metadata_filename = "spike_in/spike-in_bams_spikein_metadata.csv"
    file_dir = "./spike_in/bam"
    df = pd.read_csv(metadata_filename)    
    peak_pick_distributions(df)
    
    #bam_paths = [os.path.join(file_dir, x) for x in os.listdir(file_dir) if "calmd" in x]
    #calmd_files(bam_pahs)
    #process("./bam/test.sorted.calmd.bam")
    #create the json metadata in parallel
    #results = Parallel(n_jobs=20)(delayed(process)(bam) for bam in bam_paths[1:])
    #print(results)    

def consensus_call(target_depths, all_depths):
    #defines the spike + amino acid location to check on mutations
    start_nuc = 21563 + (677*3)
    total = max(int(k) for k, v in all_depths.items())    
    consensus_string = [''] * total
    final_con = '' 
    for key, value in all_depths.items():
        
        #if start_nuc - 4 < int(key) < start_nuc:
        #    print(key, value, target_depths[key])
    
        if key in target_depths:
            allele_dict = target_depths[key]['allele']
            total_count = target_depths[key]['total_depth']
            #if total_count < 1:
            #    best_choice = find_best_allele(all_depths, key)
            #    consensus_string += best_choice
            #    continue
            max_count = 0
            for x,y in allele_dict.items():
                if y['count'] > max_count:
                    best_choice = x
                    max_count = y['count']
            if max_count/total_count > 0.5:
                consensus_string[int(key)-1] += best_choice
                continue
            else:
                best_choice = find_best_allele(all_depths, key)
                consensus_string[int(key)-1] += best_choice
                continue
        else:
            best_choice = find_best_allele(all_depths, key)
             
            consensus_string[int(key)-1] += best_choice
                
            #if int(key) < 740 and int(key) > 728:
            #    print(int(key)+2, key,value, "\n")
            #    print(best_choice, "\n")
            continue
    
    for item in range(0,total):
        val = consensus_string[item]
        if val == '':
            final_con += 'N'
        else:
            final_con += val
    #print(final_con)
    return(final_con)     

def find_best_allele(depths, key):
        max_count = 0
        allele_dict = depths[key]['allele']
        total_count = depths[key]['total_depth']
        max_count = 0
        for x,y in allele_dict.items():
            if y['count'] > max_count:
                best_choice = x
                max_count = y['count']
        if max_count/total_count > 0.5:
            return(best_choice)
        else:
            return('N')

def peak_pick_distributions(dfh):
    """
    Parallel processing peak-picking, calling consensus, and running nextclade.
    """
    location = "./spike_in/json"
    dist_files = [os.path.join(location, item) for item in os.listdir(location) if item.endswith('.json')]
    #Parallel(n_jobs=20)(delayed(process_2)(filename) for filename in dist_files)
    process_2("./spike_in/json/add_info_file_352.json")

def process_2(filename):
    location = "./spike_in/json"
   
    print("Begin working on: ", filename)
    base_filename = filename.split("/")[-1].split(".")[0].split("_")[-2:]
    base_filename = "_".join(base_filename)

    #check and see if we have a dir for this file
    if os.path.isdir(base_filename):
        pass
    else:
        os.system("mkdir %s" %base_filename)
    """
    #select a specific file to look at
    if "file_352" in base_filename:
        pass
    else:
        continue
    """
    #open up the saved data related to read probabilties
    with open(filename, 'r') as jfile:
        data = json.load(jfile)
        probs = data['read_probs']

    #reads that have low depth or variance so we don't care about them
    #ie. zero reads gotta go!
    common_reads = [count for count,item in enumerate(probs) if item == 0]
    filtered_probs = [item for item in probs if item > 0]
    
    #makes kde function
    np.random.seed(0)
    df = pd.DataFrame(filtered_probs, columns=['data'])

    # non-parametric pdf
    nparam_density = stats.kde.gaussian_kde(filtered_probs)
    x = np.linspace(0, 1, 200)
    nparam_density = nparam_density(x)

    peaks = signal.find_peaks(nparam_density, width=0.025, height=0)
    
    
    #I pulled these out for earlier analysis I did
    indices = peaks[0]
    dictionaries = peaks[1]
    heights = dictionaries['peak_heights']
    width = dictionaries['widths']
    prominance = dictionaries['prominences']
    wh_list = dictionaries['width_heights']
    final_indices = []
    final_indices_3 = []
    ab =sorted(zip(indices, heights, width, prominance, wh_list), key=lambda x: x[1], reverse=True)
   
    #create filename for match
    filen = filename.split('/')[-1]
    filen = filen.split('.')[0].replace("add_info_","")
    
    one_off_explanation = []
    for index, h, w,p, wh in ab:
        one_off_explanation.append(index/200)

    #dump some data about our peaks to a json, prior to combining peaks
    saved_peak_data = {}
    saved_peak_data['width-heights'] = list(wh_list)
    saved_peak_data['peak-value'] = one_off_explanation
    print(saved_peak_data)
    with open("./%s/peak_information.json" %base_filename, "w") as jfile:
        json.dump(saved_peak_data, jfile)

    one_off_explanation = [0.17] 
    #lets make a blank dict with prob as key and list of counts as value
    read_classification = {}
    for thing in one_off_explanation:
        read_classification[thing] = []
     
    all_useable = 0       
    
    #iterate all probs and file the closest signal
    for count,z in enumerate(probs): 
        #useless
        if z == 0:
            continue
        y = min(one_off_explanation, key=lambda x:abs(x-z))         
        
        #this defines the window of what we call consensus on!
        if y-0.03< z < y+0.03:
            read_classification[y].append(count)            

    '''there's likely a better way to do this, but here I combine close peaks and label
    it using the high one'''
    #figure out whats close together
    combo_dict = {}
    for item in one_off_explanation:
        for x in one_off_explanation:
            if item == x:
                continue
            #same window for consensus call we saw earlier
            if item-0.03 < x < item+0.03:
                if item in combo_dict:
                    combo_dict[item].append(x)
                else:
                    combo_dict[item] = [x]
                #we can no longer use this as a peak
                one_off_explanation.remove(x)

    read_classification_2 = {}
    #combine things
    seen = []
    for key, value in read_classification.items():
        #we have two peaks that can be combined
        if key in combo_dict:
            combos = combo_dict[key]
            for thing in combos:
                value.extend(read_classification[thing])
                if thing not in seen:
                    read_classification_2[key] = value
                    seen.append(thing)
        elif key not in seen:
            read_classification_2[key] = value
    print(read_classification_2.keys())
    #sys.exit(0)
    #call consensus on whatever remains       
    for key, value in read_classification_2.items():
        process("./spike_in/bam/"+ filen + '.calmd.bam')
        sys.exit(0)
        
        print("peak: ", key, " num reads: ",len(value))
        only_peak_reads = copy.deepcopy(value)
      
        #if we need to calculate a new target depth
        new_filename_targ = './%s/extra_position_depths_%s_target.json'%(base_filename, key)
        new_filename_all = './%s/position_depths_all.json' %base_filename
        if not os.path.isfile(new_filename_targ):                
            position_depths_target = actually_call_pos_depths(only_peak_reads, "./spike_in/bam/"+ filen + ".calmd.bam")
            with open(new_filename_targ, 'w') as jfile:
                json.dump({'position_depths_target':position_depths_target}, jfile) 
        else:
            with open(new_filename_targ, 'r') as jfile:
                data = json.load(jfile)
            position_depths_target = data['position_depths_target']
        if not os.path.isfile(new_filename_all):
            position_depths_all = calculate_positional_depths("./spike_in/bam/" + filen + ".calmd.bam")
            with open(new_filename_all, 'w') as jfile:
                json.dump({'position_depths_all':position_depths_all}, jfile)
        else:
            with open(new_filename_all, 'r') as jfile:
                data = json.load(jfile)
            position_depths_all = data['position_depths_all']
            
        consensus_string = consensus_call(position_depths_target, position_depths_all)
        with open("./%s/extra_consensus_"%base_filename + filen + "_" + str(key) + ".fasta", 'w') as ffile:
            ffile.write(">%s_%s_header\n" %(filen, key))
            ffile.write(consensus_string)
            ffile.write("\n")

        #cat it on to the master file
        cmd = "cat ./%s/extra_consensus_%s_%s.fasta >> ./%s/extra_finalfile.fasta" %(base_filename, filen, str(key), base_filename)
        print(cmd)
        os.system(cmd)          

    #run nextflow on all fasta file
    cmd = "nextclade --in-order \
            --input-fasta ./%s/extra_finalfile.fasta \
            --input-dataset data/sars-cov-2 \
            --output-tsv ./%s/extra_nextclade.tsv \
            --output-tree ./%s/extra_nextclade.auspice.json \
            --output-dir ./%s \
            --output-basename extra_nextclade" %(base_filename, base_filename, \
            base_filename, base_filename)
    
    os.system(cmd)

def make_figures(probs, loc_param, scale_param, filename):
    """
    This is the code that was used to make the probability figures.
    """
    # parametric fit: assume normal distribution
    loc_param, scale_param = stats.norm.fit(probs)
    param_density = stats.norm.pdf(x, loc=loc_param, scale=scale_param)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(probs, bins=100, density=True)
    ax.plot(x, nparam_density, 'r-', label='non-parametric density (smoothed by Gaussian kernel)')
    ax.plot(x, param_density, 'k--', label='parametric density')
    ax.set_ylim([0, 10])
    ax.legend(loc='best')
    save_name = filename.split(".")
    save_name = save_name[:-1]
    save_name = '.'.join(save_name) + "_smoothing"
    plt.savefig("%s.jpg"%save_name)
    plt.close()

def process(bam):
    name = bam.split("/")[-1].split(".")[0]
    threshold = 0.0
    create_new_hdf5 = True
    process_bam_file(bam, create_new_hdf5, name, threshold)
    return(name + " done")
    
def download_files(df):
    #find all the unique combos of strains/abundances
    combo_var_abun = []
    combo_filenames = []
    for index, row in df.iterrows():
        try:
            ab = [str(x) for x in ast.literal_eval(row['abundance(%)'])]
            var = [str(x) for x in ast.literal_eval(row['variant'])]    
            ab.extend(var)
            temp_ab = "_".join(ab)
            combo_var_abun.append(temp_ab)
            combo_filenames.append(row['filename'])
        except:
            pass
     
    unique_combos, unique_indices = np.unique(combo_var_abun, return_index=True)
    unique_filenames = [x for count,x in enumerate(combo_filenames) if count in unique_indices]
    base_string = 'gsutil -m cp \\'
    gs_string = 'gs://search-reference_data/spike-in_bams/'
    
    for filename in unique_filenames:
        temp_string =  gs_string + filename + ' \\' + "\n"
        base_string += temp_string
    base_string += "." 
    os.system(base_string)

def calmd_files(file_list):
    base_command = "samtools calmd -b "
    for filename in file_list:
        output_temp = filename.split('.')
        output_temp.insert(-1,'calmd')
        output_name = ".".join(output_temp)
        cmd = base_command + filename + " sequence.fasta > " + output_name       
        os.system(cmd)
 
if __name__ == "__main__":
    main()    
