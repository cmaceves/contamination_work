import os
import sys
import ast
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import signal
#other script import
from contaminant_analysis import calculate_positional_depths, calculate_read_probability, call_pos_depths_read, \
    actually_call_pos_depths

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
   
    start_nuc = 21563 + (478*3)
    total = max(int(k) for k, v in all_depths.items())    
    consensus_string = [''] * total
    final_con = '' 
    for key, value in all_depths.items():
        
        #if start_nuc - 4 < int(key) < start_nuc:
        #    print(key, value,)
    
        if key in target_depths:
            if int(key) == 23604:
                print(value)
                print(target_depths[key])
            #if int(key) == 23403:
            #    print(value)
            #    print(target_depths[key])
            #if int(key) == 21618:
                #print("delta ", target_depths[key])
                #print(value)
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
    from scipy import stats

    location = "./spike_in/json"
    dist_files = [os.path.join(location, item) for item in os.listdir(location) if item.endswith('.json')]
    new_dict = {'filename':[], 'percent':[], "percent_score":[]}
    for filename in dist_files:
        print(filename)
        if "file_12." not in filename:
            continue
        
        with open(filename, 'r') as jfile:
            data = json.load(jfile)
            probs = data['read_probs']
        
        save_probs = probs
        #reads that have low depth or variance so we don't care about them
        common_reads = [count for count,item in enumerate(probs) if item == 0]
        probs = [item for item in probs if item > 0]
        
        #makes kde function
        np.random.seed(0)
        df = pd.DataFrame(probs, columns=['data'])

        #mut at aa 19
        delta_read = [562185, 562192, 562207]

        # non-parametric pdf
        nparam_density = stats.kde.gaussian_kde(probs)
        x = np.linspace(0, 1, 200)
        nparam_density = nparam_density(x)

        peaks = signal.find_peaks(nparam_density, width=0.025, height=0)
        
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
        new_dict['filename'].append(filen + '.bam')

        #get ground truth
        peak_expected_at = dfh.loc[dfh['filename'] == filen +'.bam'].values[0][-1]
       
        one_off_explanation = []
        one_off_score = []
        for index, h, w,p, wh in ab:
            one_off_explanation.append(index/200)
            one_off_score.append(wh)

        new_dict['percent_score'].append(one_off_score)
        new_dict['percent'].append(one_off_explanation)
        
        read_classification = {}
        for thing in one_off_explanation:
            read_classification[thing] = []
        
        all_useable = 0       
        #iterate all probs and file the closest signal
        for count,z in enumerate(save_probs): 
            if z == 0:
                continue
            y = min(one_off_explanation, key=lambda x:abs(x-z))         
            
            if y-0.05< z < y+0.05:
                read_classification[y].append(count)            
 
        print(one_off_explanation)
        print("all reads: ", len(save_probs))
        print("common reads: ", len(common_reads))
        new_all = True        
        #call consensus on whatever remains       
        for key, value in read_classification.items():
            #process("./spike_in/bam/"+ filen + '.calmd.bam')
            #sys.exit(0)

            print("peak: ", key, " num reads: ",len(value))
            import copy
            only_peak_reads = copy.deepcopy(value)
            print(len(only_peak_reads))
            print("should remove: ", len(save_probs) - len(common_reads) - len(value))
            value.extend(common_reads)
            removal_reads = set(range(0,len(save_probs))) - set(value)
            removal_reads = list(removal_reads)
            print("remove: ", len(removal_reads))
            removal_filename = filen + "_" + str(key) + "_remove.txt"
            #print("saving removal reads to: ", removal_filename)
            #call_pos_depths_read(removal_reads, "./spike_in/bam/"+ filen + '.calmd.bam', removal_filename)
            new_calc = True
            
            if new_calc:                
                position_depths_target = actually_call_pos_depths(only_peak_reads, "./spike_in/bam/"+ filen + ".calmd.bam")
                with open('position_depths_target.json', 'w') as jfile:
                    json.dump({'position_depths_target':position_depths_target}, jfile) 
            else:
                with open('position_depths_target.json', 'r') as jfile:
                    data = json.load(jfile)
                position_depths_target = data['position_depths_target']
            if new_all: 
                new_all = False
                position_depths_all = calculate_positional_depths("./spike_in/bam/" + filen + ".calmd.bam")
                with open('position_depths_all.json', 'w') as jfile:
                    json.dump({'position_depths_all':position_depths_all}, jfile)
            else:
                with open('position_depths_all.json', 'r') as jfile:
                    data = json.load(jfile)
                position_depths_all = data['position_depths_all']
                
            consensus_string = consensus_call(position_depths_target, position_depths_all)
            with open("consensus_" + filen + "_" + str(key) + ".fasta", 'w') as ffile:
                ffile.write(">%s\n" %filen)
                ffile.write(consensus_string)  
        sys.exit(0)
        continue

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

    df = pd.DataFrame(new_dict)
    df2 = df.merge(dfh, on="filename")
    print(df2)
    df2.to_csv("peak_pick.csv")

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
