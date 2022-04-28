from joblib import Parallel, delayed
from scipy import signal
from scipy import stats
import os
import sys
import math
import json
import pickle
from sklearn.utils import shuffle
import pysam
from itertools import permutations
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy import linalg
import matplotlib as mpl

from sklearn import mixture
from contaminant_analysis import calculate_positional_depths, calculate_read_probability, \
    actually_call_pos_depths
from process_spike_in import consensus_call

def parallel():
    metadata_filename = "spike_in/spike-in_bams_spikein_metadata.csv"
    df = pd.read_csv(metadata_filename)
    
    file_nums_all = [a.split('.')[0].split('_')[-1] for a in list(df['filename'])]
    #file_nums_all = file_nums_all[:1]
    results = Parallel(n_jobs=20)(delayed(process)(file_num) for file_num in file_nums_all)
    
def process(file_num):
    if os.path.isfile('cluster_%s.json' %file_num):
        return(0)
    print('working on %s' %file_num)
    
    filename = "./spike_in/bam/file_%s.calmd.bam" %file_num
    if os.path.isfile(filename):
        pass
    else:
        return(-1)
    pos_filename = '../contamination_work/file_%s.json' %file_num
    hdf5_filename = 'myfile_file_%s.hdf5' %file_num
    if os.path.isfile(pos_filename): 
        with open('pos_trial_%s.json' %file_num, 'r') as jfile:
            pos_depths = json.load(jfile)
    else:
        pos_depths = calculate_positional_depths(filename)
        with open('pos_trial_%s.json' %file_num, 'w') as jfile:
            json.dump(pos_depths, jfile)
    if os.path.isfile(hdf5_filename):
        pass
    else:
        calculate_read_probability(pos_depths, filename, 'file_%s' %file_num, 0, True)    
    return(1) 
    probs = []
    count_a = []
    #first we list the informative positions
    for key, value in pos_depths.items():
        """
        if int(key) > 21623-4 and int(key) < 21623:
            print(key,value)
        """
        total_depth = value['total_depth']
        if total_depth < 10:
            continue
        ref = max(value['ref'], key=value['ref'].get)
        allele_dict = value['allele']
        useful_alleles = []
        for k,v in allele_dict.items():
            if k == ref:
                continue
            if k == 'N':
                continue
            if v['count']/total_depth < 0.03:
                continue
            useful_alleles.append(k)
        if len(useful_alleles) >= 1:
            pass
        else:
            continue
        count_a.append(key)
    
    indexes = [int(item) for item in count_a] 
    
    with open("add_info_file_%s.json" %file_num, "r") as afile:
        data =json.load(afile)
    read_probs = data['read_probs'] 
    X=[]
    ioi = []
    if os.path.isfile('ioi_%s.json' %file_num) and os.path.isfile('probs_%s.npy'%file_num):
        with open('ioi_%s.json' %file_num, 'r') as jfile:
            ioi = json.load(jfile)['ioi']
        X = np.load("probs_%s.npy" %file_num)
        X = np.array(X) 
    
    else:
        with h5py.File('myfile_file_%s.hdf5' %file_num, 'r') as hfile:
            dset = hfile['default']
            for count,(row,rprobs) in enumerate(zip(dset,read_probs)):
                if count % 100000 == 0:
                    #print(count)            
                    pass
                if rprobs > 0.03:
                    #pull indexes of interest
                    temp = row[indexes]
                    if np.count_nonzero(temp) == 0:
                        continue
                    X.append(temp)
                    ioi.append(count)
                 
        X = np.array(X)
        np.save("./probs_%s.npy" %file_num, X)
        
        with open('ioi_%s.json' %file_num, 'w') as jfile:
            json.dump({'ioi':ioi}, jfile)
        
    encoded_alleles = {"A":1, "C":2, "G":3, "T":4, "N":5}
    
    if os.path.isfile('allele_matrix_%s.npy'%file_num):
        alleles = np.load("allele_matrix_%s.npy" %file_num)
    else:
        #now we go back and get the actual alleles
        samfile = pysam.AlignmentFile(filename, "rb")
        alleles = np.zeros((X.shape[0], X.shape[1]))
        for count, thing in enumerate(samfile):
            if count % 100000 == 0:
                #print(count)
                pass
            if count in ioi:
                pass
            else:
                continue
            allele_temp = []
            for (pos, nuc, ref, qual) in zip(thing.get_reference_positions(), \
                thing.query_alignment_sequence, thing.get_reference_sequence(), thing.query_alignment_qualities):
                if str(pos+1) in count_a:
                    alleles[ioi.index(count)][int(count_a.index(str(pos+1)))] = encoded_alleles[nuc]
                        
        np.save("./allele_matrix_%s.npy"%file_num, alleles) 
       
   
    #should be the beta mutation at S gene 80 ie. 21801 A->C    
    #should be the alpha mutation at S gene 570 ie. 23271 X->A
    #should be the alpha mutation at S gene 982 ie. 24506 T->G
    #should be the delta mutation at S gene 19 ie. 21618 X->X 
    #should be the delta mutation at S gene 950 ie. 24410 G->A
    #should be the gamma mutation at S gene 20 ie. 21621 C->A

    new = np.dstack((X,alleles))    

    """   
    extract = new[:,indexes.index(21618),:]
    extract2 = new[:, indexes.index(24410),:]
    a_count = 0
    c_count = 0
    t_count = 0
    g_count = 0 
     
    print("delta index")
    delta_index = []
    for count,(thing,thing2) in enumerate(zip(extract, extract2)):
        #mut at S19
        if thing[0] != 0:
            tol = thing[1]
            if tol == 1:
                a_count += 1
            elif tol == 2:
                c_count += 1
            elif tol == 3:
                g_count += 1
                delta_index.append(count)
            elif tol == 4:
                t_count += 1
        #mut at S950
        if thing2[0] != 0:
            tol = thing[1]
            if tol == 1:
                delta_index.append(count)
        
    delta_index = list(np.unique(delta_index))
    
    #print(a_count, " ", c_count, " ", g_count , " ", t_count )    
    
    extract = new[:,indexes.index(23271),:]
    extract2 = new[:, indexes.index(24506), :]
    a_count = 0
    c_count = 0
    t_count = 0
    g_count = 0 
    
    print("alpha index")
    alpha_index = []
    for count,(thing,thing2) in enumerate(zip(extract,extract2)):
        if thing[0] != 0:
            tol = thing[1]
            if tol == 1:
                a_count += 1
                alpha_index.append(count)
                first = count
            elif tol == 2:
                c_count += 1
            elif tol == 3:
                g_count += 1
            elif tol == 4:
                t_count += 1
        if thing2[0] != 0:
            tol = thing2[1]
            if tol == 3:
                second = count
                alpha_index.append(count)

    #print(a_count, " ", c_count, " ", g_count , " ", t_count )    
        
    extract = new[:,indexes.index(21621),:]
    a_count = 0
    c_count = 0
    t_count = 0
    g_count = 0 
    print("gamma index") 
    gamma_index = []
    for count,thing in enumerate(extract):
        if thing[0] != 0:
            tol = thing[1]
            if tol == 1:
                a_count += 1
                gamma_index.append(count)
            elif tol == 2:
                c_count += 1
            elif tol == 3:
                g_count += 1
            elif tol == 4:
                t_count += 1
    
    extract = new[:,indexes.index(21801),:]
    a_count = 0
    c_count = 0
    t_count = 0
    g_count = 0 
    print("beta index") 
    beta_index = []
    for count,thing in enumerate(extract):
        if thing[0] != 0:
            tol = thing[1]
            if tol == 1:
                a_count += 1
            elif tol == 2:
                c_count += 1
                beta_index.append(count)
            elif tol == 3:
                g_count += 1
            elif tol == 4:
                t_count += 1
    """
    
    """
    g1 = data['g_1']['i']
    g3 = data['g_3']['i']
    g6 = data['g_6']['i']
    g8 = data['g_8']['i']
    g9 =data['g_9']['i']
    g11 = data['g_11']['i']
    g22 = data['g_22']['i'] 
    
    extract = new[:,indexes.index(22995),:]
    for c,test in enumerate(extract):
        if test[0] <= 0:
            continue
        if c in g1:
            #pass
            print('g1', test)
        if c in g3:
            pass
            #print('g3', test)
        if c in g6:
            #pass
            print('g6', test)
        if c in g8:
            #pass
            print('g8', test)
        if c in g9:
            print('g9', test)
        if c in g11:
            #pass
            print('g11', test)
        if c in g22:
            #pass
            print('g22', test)
    sys.exit(0)  
    """

    if os.path.isfile('cluster_%s.json' %file_num):
        pass 
    else:     
        total_pos = new.shape[1]
        total_cov = []
        store_matrix = np.zeros((new.shape[0], new.shape[0]))
        
        #part where we actually group things up
        gl = {'g_1':{'i':[], 'p':[] }}
        
        for c, x in enumerate(new):
            #just keeping track of things
            if c % 100000 == 0:
                #print(c)
                pass
            if c == 0:
                gl['g_1']['i'].append(c)
                
            present_locs = [a for a,b in enumerate(x[:,0]) if b > -1]
            present_probs = [b for b in x[:,0] if b > -1]
            present_probs = [b if b > 0.03 else 0 for b in present_probs]
            avg_snv_peakos = [b for b in present_probs if b > 0]
            avg_snv = np.average(avg_snv_peakos)
            present_nucs = [b for a,b in enumerate(x[:,1]) if a in present_locs]

            peaks = avg_snv_peakos
            # non-parametric pdf
            if len(avg_snv_peakos) > 1:
                try: 
                    x = np.linspace(0, 1, 200)
                    nparam_density = stats.kde.gaussian_kde(avg_snv_peakos)
                    nparam_density = nparam_density(x)
                    peaks = signal.find_peaks(nparam_density, width=0.025, height=0)
                except:
                    print("LIN SPACE FAIL")
                    continue
                peaks = [a/200 for a in peaks[0]]
            if c == 0:
                gl['g_1']['p'].append(peaks)
                          
            best_group = ''
            best_val = 0

            #we try and find a close match in the last several items
            for key,value in gl.items():
                #value = value['i']
                value = value['p']            
                for i in range(0,1): 
                    if best_val == 1:
                        break
                    comp_i = 0
                    past = value[i]
                    for thing1 in peaks:
                        for thing2 in past:
                            if thing1-0.01 < thing2 < thing1+0.01:
                                comp_i += 1
                    if comp_i > 0 and comp_i/len(peaks) > best_val:
                        best_val = comp_i/len(peaks)
                        best_group=key
            if best_val <= 0.5 and len(peaks) > 0:
                last_key = list(gl.keys())[-1].split("_")[-1]
                new_key = "g_" + str(int(last_key)+1)
                for k,v in gl.items():
                    print(k, len(v['i']), v['p'][-1])
                gl[new_key] = {'i':[c], 'p':[peaks]} 
                continue
            if best_group == '':
                continue
            gl[best_group]['i'].append(c)
            gl[best_group]['p'].append(peaks)
        
        for key, value in gl.items():
            print(key, len(value['p']))
        with open('cluster_%s.json' %file_num, 'w') as jfile:
            json.dump(gl, jfile)
    
    print("post hoc on ", file_num)    
    #figure out how many groups are in the sample
    post_hoc(file_num, new, indexes, ioi, filename, pos_depths)
    #remove_str = 'myfile_file_%s.hdf5' %file_num
    #os.system('rm %s' %remove_str)
    return("0")

def pull_reads_cluster(indices, bam, cluster_num, all_depths, file_num):
    """
    Wraps two functions, (1) calling pos depths on certain reads
    and (2) calling consensus on pos reads into one and output it
    into a cluster specific file.
    """
    print("pulling reads to a consensus")

    target_depths = actually_call_pos_depths(indices, bam)
    consensus_string = consensus_call(target_depths, all_depths)           
    
    with open("./extra_consensus_%s_%s.fasta" %(file_num, cluster_num), 'w') as ffile:
        ffile.write(">%s_header\n" %(cluster_num))
        ffile.write(consensus_string)
        ffile.write("\n")

def find_closest_sum(numbers, target, n):
    permlist = list(permutations(numbers, n))
    sumlist = [sum(l) for l in permlist]
    
    maxpos = 0
    for i in range(1, len(sumlist)):
        if abs(sumlist[i] - target) < abs(sumlist[maxpos]-target):
             maxpos = i

    return permlist[maxpos]

def print_group_locations(data,alpha_index,gamma_index, delta_index):
    """
    Find the group that the "known" indices fell into.
    """
    for key, value in data.items():
        val1 = value['i']
        tmp = [a for a in alpha_index if a in val1]
        print('alpha', key, len(tmp))
        tmp = [a for a in gamma_index if a in val1]
        print('gamma', key, len(tmp))
        #tmp = [a for a in beta_index if a in val1]
        #print('beta', key, len(tmp))
        tmp = [a for a in delta_index if a in val1]
        print('delta', key, len(tmp))

def post_hoc(file_num, new, indexes, ioi, filename, pos_depths):
    with open("cluster_%s.json"%file_num, "r") as jfile:
        data = json.load(jfile)
    
    #let's see if we can recover the correct things
    #gather the appropriate indices
    actual_index_dict = {}
    for key, value in data.items():
        val1 = value['i']
        
        #remove things with small number of reads, remove things with high prob
        #if len(val1) > 3000 and value['p'][0][0] < 0.97:
            #we care about this group
            #print(key, len(value['i']), value['p'][0])
        actual_index_dict[key] =[]
        for v in val1:              
            actual_index_dict[key].append(v)

    #what are the most common mutations in each group?
    mut_locs_dict = {}
    all_m = []
    
    for k,v in actual_index_dict.items():
        mut_locs = []
        for item in v:
            x = new[item,:,0]
            present_locs = [a for a,b in enumerate(x) if b > 0]
            mut_locs.extend(present_locs)
        muts, counts = np.unique(mut_locs, return_counts=True)

        muts = [x for _, x in sorted(zip(counts, muts), reverse=True)]
        counts = [b for b, x in sorted(zip(counts, muts), reverse=True)]
        
        #we want to take mutations associated with 99% of the reads
        num_reads = len(data[k]['p'])*0.99
        tc=0
        for m,c in enumerate(counts):
            tc+=c
            if tc >= num_reads:
                break    
         
        act_loc = [indexes[a]  for a in muts]
        mut_locs_dict[k] = act_loc[:m+1]
        all_m.extend(act_loc[:m+1])
        print(file_num, k, act_loc[:m+1], data[k]['p'][0])
      
    
    #for things that fall under the peaks
    lookat = [(266,21555), (21563, 25384), (26245, 26472), (26245, 26472), (26523, 27191), \
    (27202, 27387), (27394, 27759), (27756, 27887), (27894, 28259), (28274, 29533), (29558,29674)] 
    pop = []
    #look for a mutation we think we see occuring in     
 
    for group in lookat:
        t = []
        for k,v in mut_locs_dict.items():
            for l in v:
                if l > group[0] and l < group[1]:
                    t.append(k)
        pop.append(t)
    t_group = list(np.unique(pop[0]))
    pooled_muts = []
    t_dict = {}
    print(mut_locs_dict)
    for thing in t_group:
        t_dict[thing] = {'mut':[a for a in mut_locs_dict[thing]], 'freq':data[thing]['p'][0]}
        pooled_muts.extend([a for a in mut_locs_dict[thing]])
    pooled_muts = list(np.unique(pooled_muts))   
    
    unique_g = []
    for thing in pooled_muts:
        seen = []
        for k,v in t_dict.items():
            if thing in v['mut']:
                seen.append(k)
        if len(seen) > 1:
            continue
        else:
            unique_g.extend(seen)
    unique_g = list(np.unique(unique_g))
    freq = []
    for k,v in t_dict.items():
        if k not in unique_g:
            continue
        freq.append(v['freq'][0])
    print(freq)
    res = []
    res_2 = []
    if len(freq) > 0: 
        print('brute force permute freq') 
        if len(freq) > 1:
            result_shown = find_closest_sum(freq, 1, 2)
            #print(result_shown)
            res_2.append(result_shown)
            res.append(sum(list(result_shown)))

        if len(freq) > 2:
            result_shown = find_closest_sum(freq, 1, 3)
            #print(result_shown)
            res_2.append(result_shown)
            res.append(sum(list(result_shown)))

        if len(freq) > 3:
            result_shown = find_closest_sum(freq, 1, 4)
            res.append(sum(list(result_shown)))
            res_2.append(result_shown)
            #print(result_shown)    
        
        try:
            closest = min(res, key=lambda x:abs(1)) 
            groups = res.index(closest)
            freq_of_choice = res_2[groups]
            groups_name = [unique_g[freq.index(i)] for i in list(freq_of_choice)]
           
            df = pd.DataFrame({"filename":[file_num], "predicted_num":[len(groups_name)], \
                "groups":[str(groups_name)], "freq":[str(freq_of_choice)], "muts": [str(t_dict)]})
        except:
            df = pd.DataFrame({"filename":[file_num], "predicted_num":[1], \
            "groups":["None"], "freq":[freq], "muts": [str(mut_locs_dict)]})
        
    else: 
        df = pd.DataFrame({"filename":[file_num], "predicted_num":["None"], \
        "groups":["None"], "freq":["None"], "muts": [str(mut_locs_dict)]})
    
    if os.path.isfile('master_results.csv'):
        df.to_csv('master_results.csv', mode='a', header=False)
    else:
        df.to_csv('master_results.csv')
    """
    for g in groups_name:
        act_loc = [ioi[a]  for a in data[g]['i']]
        pull_reads_cluster(act_loc, filename, g, pos_depths, file_num)
    """

if __name__ ==  "__main__":
    parallel()
