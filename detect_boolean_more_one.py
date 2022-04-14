"""
Attempt to determine
(1) Whether we have 1+ things in a file
(2) Given that information, can we call consensus?
(3) Can we recommend a threshold?
"""
import os
import sys
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from contaminant_analysis import calculate_positional_depths, calculate_read_probability, \
    actually_call_pos_depths

def percent_variance(pos_depths):
    pos_var = {}
    all_var = []
    i = 0
    for key, value in pos_depths.items():
        if i % 100000 == 0:
            print(i)
        i += 1
        exp_alleles = value['allele']
        ref_alleles = value['ref']
        total_count = value['total_depth']        
        
        ref = max(ref_alleles, key=ref_alleles.get)         
       
        #calc freq for every allele not matching the ref
        for akey, aval in exp_alleles.items():
            #if matches ref, we don't care
            if akey == ref:
                continue 
            #if the depth < 10 we also don't care
            if aval['count'] < 10:
                continue

            #calculate frequency
            percent_var = aval['count']/total_count
            
            #if our frequency is especially low we discount it
            if percent_var < 0.03:
                continue

            all_var.append(percent_var)
            if key in pos_var:
                pos_var[key][akey] = percent_var
            else:
                pos_var[key] = {akey : percent_var}
    
    
    bw = bestBandwidth(all_var)    
    data = np.array(all_var).reshape(-1, 1)
    
    print(all_var)
    """
    print(len(all_var))
    kde = KernelDensity(kernel = 'gaussian', bandwidth=bw).fit(data)
    clustering = MeanShift(bandwidth=bw).fit(kde)    
    print(clustering)
    sys.exit(0)
    """
    clustering = MeanShift(bandwidth=bw).fit(data)
    print(clustering.labels_)
    print(np.unique(clustering.labels_, return_counts=True))

    #plotting
    sns.kdeplot(all_var, bw_method=bw)
    plt.title("KDE of SNV frequency 5/95% mix " + str(round(bw,2)))
    plt.savefig('all_var_2.png')       

def bestBandwidth(data, minBandwidth = 0.01, maxBandwidth = 1, nb_bandwidths = 30, cv = 30):
    """
    Run a cross validation grid search to identify the optimal bandwidth for the kernel density
    estimation.
    """
    data = np.array(data).reshape(-1, 1)
    model = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(minBandwidth, maxBandwidth, nb_bandwidths)}, cv=cv) 
    model.fit(data)

    return model.best_params_['bandwidth']
  
def main():
    #1 thing delta
    filename_1 = "./spike_in/bam/file_324.calmd.bam"
    #2 things, can call consensus
    filename_2 = "./spike_in/bam/file_0.calmd.bam"
    #2 things, can't call consensus
    filename_3 = "./spike_in/bam/file_260.calmd.bam"
    #more than 2 things, can't call consensus
    filename_3 = "./spike_in/bam/file_364.calmd.bam"

    #do we think we have more than one thing in the file?
        

    #get all depths
    if os.path.isfile("test_2.json"):
        with open("test_2.json", "r") as jfile:
            pos_depths = json.load(jfile) 
    else:
        pos_depths = calculate_positional_depths(filename_2)     
        with open("test_2.json",'w') as jfile:
            json.dump(pos_depths, jfile)

    #all percent variances at all positions
    percent_variance(pos_depths) 


if __name__ == "__main__":
    main()
