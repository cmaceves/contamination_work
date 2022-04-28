import os
import math
import copy
import time
import sys
import statistics
import pandas as pd
import numpy as np
import seaborn as sns
import pysam
import pickle
import json
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def actually_call_pos_depths(keep_reads, bam):
    samfile = pysam.AlignmentFile(bam, "rb")
    position_dict = {}
    query_names = []
    used_reads = 0
    print("Passed reads: ", len(keep_reads))
    #loop through all reads
    for count,thing in enumerate(samfile): 
        if count % 100000 == 0:
            print(count)
        if count not in keep_reads:
            continue
        #loop over each position in each read
        for (pos, letter, ref, qual) in zip(thing.get_reference_positions(), thing.query_alignment_sequence, thing.get_reference_sequence(), thing.query_alignment_qualities):
            #standardize this
            letter = letter.upper()
            ref = ref.upper()
            pos = str(pos + 1)

            #if we've seen this position before
            if pos in position_dict:
                #check if we've seen this nuc before
                if letter in position_dict[pos]["allele"]:
                    position_dict[pos]['allele'][letter]["count"] += 1
                    position_dict[pos]['total_depth'] += 1 
                    position_dict[pos]['allele'][letter]["qual"] += qual    
                #else we will add the letter            
                else:
                    position_dict[pos]['allele'][letter] = {"count":1, 'qual':qual}
                    position_dict[pos]['total_depth'] += 1

                #handle the ref addition
                if ref in position_dict[pos]['ref']:
                    position_dict[pos]['ref'][ref] += 1
                else:
                    position_dict[pos]['ref'][ref] = 1
                    
            #else we'll add this position
            else:
                position_dict[pos] = {}
                position_dict[pos]['ref'] = {ref:1}
                position_dict[pos]['allele'] = {letter:{"count": 1, 'qual': qual}}
                position_dict[pos]['total_depth'] = 1 
        used_reads += 1
        #if len(thing.get_reference_positions()) != len(thing.get_reference_positions(full_length=True)):
            #print(thing.query_sequence, thing.get_reference_positions(full_length=True))
            #let's ship it off to support function to find out if we have any insertions/deletions
            #position_dict = find_insertions_deletions(position_dict, \
            #    thing.get_reference_positions(full_length=True), thing.query_sequence, thing.qual)

        query_names.append(thing.query_name)
    print("Used reads: ", used_reads)
    return(position_dict)

   

def alternate_calc_pos_depths(bam, test):
    """
    Creates a dict for each positions from pileup.
    """    
    samfile = pysam.AlignmentFile(bam, "rb")
    pos = 10376
       
    for pileupcolumn in samfile.pileup("PRV", pos, pos+1):
        print(pileupcolumn.get_query_sequences(add_indels=True))
        for pile in pileupcolumn.pileups:
            print(pileupcolumn.pos, pileupcolumn.n)
            print(pile.is_del, pile.indel)
            if not pile.is_del and not pile.is_refskip:
               print(pile.alignment.query_sequence[pile.query_position])
               sys.exit(0)
#@profile
def calculate_positional_depths(bam):
    """
    Creates a dict for each position with nucs, depths, and qual.
    
    Parameters
    ----------
    bam : str
        Path to the bam file to iterate.
    
    Returns
    -------
    position_dict : dict
        Nucs with depths and qual at each position.
    """
    print("Calculating positional depths: ", bam)
    query_names = []
    samfile = pysam.AlignmentFile(bam, "rb")
      
    unique_headers = []
    position_dict = {}    
    seen_reads = [] 
    
    """
    for fullcount, pileupcolumn in enumerate(samfile.pileup("NC_045512.2")):
        if fullcount % 100 == 0 and fullcount > 0:
            print(fullcount)
            break 
        for pileupread in pileupcolumn.pileups:
            if pileupread.alignment.qname in seen_reads:
                continue
            else:
                seen_reads.append(pileupread.alignment.qname)
    """
    past_r = ''
    for fullcount, pileupread in enumerate(samfile):
        if fullcount % 100000 == 0:
            if fullcount > 0:
                pass
                #print(fullcount)
        cigar = pileupread.cigartuples
        cigar = [list(x) for x in cigar]

        start = pileupread.query_alignment_start
        end = pileupread.query_alignment_end
        pos = pileupread.get_reference_positions(full_length=True)[start:end]
        
        #if the length is different their's an insertion
        ref_seq = pileupread.get_reference_sequence()
        ref_pos = pileupread.get_reference_positions()

        seq = pileupread.query_alignment_sequence 
        total_ref = pileupread.get_reference_positions(full_length=True)
        
        insertion=''
       
        finished=False
        on_insertion=False       
        for count, (r, q, qual) in enumerate(zip(total_ref, pileupread.query_sequence, \
            pileupread.query_qualities)):
            total=0                                   

            #this tells you what it should be based on the cigar, but if it doesn't land
            #in the del range you remove prior deletiosn from the cig
            for x in cigar:
                if x[0] == 2 or x[0] == 5:
                    continue
                total += x[1]
                if total > count:
                    cigtype = x[0]
                    break
            if cigtype == 0:
                if on_insertion:
                   stored_nuc = insertion
                   past_r = total_ref[count-len(stored_nuc)-1]                      
                   #print(stored_nuc, len(stored_nuc), past_r, total_ref, count, cigar) 
                nuc_add = q.upper() 
                #print(cigtype, cigar, count, total_ref, ref_pos, r, count)
                rloc = ref_pos.index(r)
                ref = ref_seq[rloc].upper()
                finished=True
                if past_r is None:
                    past_r = r-1
            #if we have an insertion we have no reference
            elif cigtype == 1:
                on_insertion = True
                finished=False
                insertion += q.upper()                
            #if we have a deletion, the reference is the same as the nucs
            elif cigtype == 2:
                continue
            elif cigtype == 4:
                continue
            else:
                print(cigar)
                print(len(total_ref), len(pileupread.query_alignment_sequence))
                print(cigtype)
                print("cig type not accounted for")
                sys.exit(0)
            #if we aren't finished, keep iterating
            if not finished:
                continue
            if r is None or str(r) == "null":
                print(r, count, total_ref, cigar)
            #if we are finished, we add the match either way
            if not on_insertion:
                if  r not in position_dict:
                    position_dict[r] = {}
                    position_dict[r]['ref'] = {ref:1}
                    position_dict[r]['allele'] = {nuc_add:{"count": 1, 'qual': qual}}
                    position_dict[r]['total_depth'] = 1
                else: 
                    #check if we've seen this nuc before
                    if nuc_add in position_dict[r]["allele"]:
                        position_dict[r]['allele'][nuc_add]["count"] += 1
                        position_dict[r]['total_depth'] += 1 
                        position_dict[r]['allele'][nuc_add]["qual"] += qual    
                    #else we will add the letter            
                    else:
                        position_dict[r]['allele'][nuc_add] = {"count":1, 'qual':qual}
                        position_dict[r]['total_depth'] += 1
                    #handle the ref addition
                    if ref in position_dict[r]['ref']:
                        position_dict[r]['ref'][ref] += 1
                    else:
                        position_dict[r]['ref'][ref] = 1
            if on_insertion:
                if past_r is None:
                    print(stored_nuc, r, total_ref, cigar)
                r = past_r              
                #insertion at an unseen position  
                if r not in position_dict and on_insertion:
                    store_nuc = "+"+stored_nuc
                    position_dict[r] = {}
                    position_dict[r]['ref'] = {}
                    position_dict[r]['allele'] = {store_nuc:{"count": 1, 'qual': qual}}
                    position_dict[r]['total_depth'] = 1
                    on_insertion=False
                    insertion=''
                    stored_nuc=''
                #we have an insertion at a position we've seen before
                elif r in position_dict and on_insertion:
                    store_nuc = "+"+stored_nuc
                    if store_nuc in position_dict[r]["allele"]:
                        position_dict[r]['allele'][store_nuc]["count"] += 1
                        position_dict[r]['total_depth'] += 1 
                        position_dict[r]['allele'][store_nuc]["qual"] += qual    
                    #else we will add the letter            
                    else:
                        position_dict[r]['allele'][store_nuc] = {"count":1, 'qual':qual}
                        position_dict[r]['total_depth'] += 1
                    on_insertion=False
                    insertion=''
                    stored_nuc=''
 
    return(position_dict)               
    sys.exit(0)   
    for count,thing in enumerate(samfile): 
        if count % 100000 == 0:
            #print(count)
            #break
            pass
        #loop over each position in each read
        for (pos, letter, ref, qual) in zip(thing.get_reference_positions(), thing.query_alignment_sequence, thing.get_reference_sequence(), thing.query_alignment_qualities):
            #standardize this
            letter = letter.upper()
            ref = ref.upper()
            pos = str(pos + 1)

            #if we've seen this position before
            if pos in position_dict:
                #check if we've seen this nuc before
                if letter in position_dict[pos]["allele"]:
                    position_dict[pos]['allele'][letter]["count"] += 1
                    position_dict[pos]['total_depth'] += 1 
                    position_dict[pos]['allele'][letter]["qual"] += qual    
                #else we will add the letter            
                else:
                    position_dict[pos]['allele'][letter] = {"count":1, 'qual':qual}
                    position_dict[pos]['total_depth'] += 1

                #handle the ref addition
                if ref in position_dict[pos]['ref']:
                    position_dict[pos]['ref'][ref] += 1
                else:
                    position_dict[pos]['ref'][ref] = 1
                    
            #else we'll add this position
            else:
                position_dict[pos] = {}
                position_dict[pos]['ref'] = {ref:1}
                position_dict[pos]['allele'] = {letter:{"count": 1, 'qual': qual}}
                position_dict[pos]['total_depth'] = 1 
   
    return(position_dict)

def find_insertions_deletions(position_dict, reference_positions, query_sequence, query_qual):
    """
    Finds any insertions/deletions and adds them to the pos dict.
    
    Parameters
    ----------
    position_dict : dict
        
    reference_positions : iterable
        Full positions including insertions/deletions for read.
    query_sequence : iterable
        Full query sequence inclusing insertions/deletions for read.
    query_qual : iterable
        The qualities for the full query.

    Returns
    -------
    position_dict : dict
    """
    
    total_length = len(query_sequence)
    insertion_starts = None
    insertion_letters = ""
    
    #iterate looking for Nones
    for count, (pos, letter, qual) in enumerate(zip(reference_positions, query_sequence, query_qual)):
        
        #we have an insertion and it's the first letter
        if pos is None and insertion_starts is None:
            #if the insertion is the not the first thing in the read
            if count != 0 :
                #find where the insertion begins
                insertion_starts = reference_positions[count-1]
                try:
                    insertion_letters += letter
                except:
                    print("FAILED", letter, reference_positions, query_sequence)
                    sys.exit(0)
            #if the insertion is the first thing in the read
            else:
                insertion_starts = count
                insertion_letters += letter

        #we continue to add to the insertion
        if pos is None and insertion_starts is not None:
            insertion_letters += letter    
            
        #we've found the entire insertion, we add it to the pos dict and reset                
        if pos is not None and insertion_starts is not None:
            if insertion_starts == 0:
                insertion_starts = pos - 1
            insertion_letters = '+'+insertion_letters
            #we've seen this position
            if insertion_starts in position_dict:
                if insertion_letters in position_dict[insertion_starts]['allele']:
                    position_dict[insertion_starts]['allele'][insertion_letters]["count"] += 1
                    position_dict[insertion_starts]['total_depth'] += 1 
                    position_dict[insertion_starts]['allele'][insertion_letters]["qual"]=None 
                #else we will add the letter            
                else:
                    position_dict[insertion_starts]['allele'][insertion_letters] = \
                        {"count":1, 'qual':None}
                    position_dict[insertion_starts]['total_depth'] += 1
            else:
                position_dict[insertion_starts] = {}
                position_dict[insertion_starts]['ref'] = {}
                position_dict[insertion_starts]['allele'] = \
                    {insertion_letters:{"count": 1, 'qual': None}}
                position_dict[insertion_starts]['total_depth'] = 1 
                
            if insertion_starts < 123:                 
                print(insertion_starts, insertion_letters)
                
            insertion_starts = None
            insertion_letters = ""
    
    return(position_dict)

def calculate_read_probability(position_depths, bam, name, \
    threshold, create_new_hdf5=False, virus_1=None, virus_2=None, gt_dict=None, pic_name=None):
    print("Begin calculating read probabilties: ", bam)
    
    samfile = pysam.AlignmentFile(bam, "rb") 
    
    if create_new_hdf5:
        f = h5py.File('myfile_%s.hdf5' %name,'w')
        #if not log_letters:
        dset = f.create_dataset("default", (1,30000), maxshape=(2700000,30000), chunks=True, compression="gzip")
        #we mark the read we think have "contam" nucs
        contam_reads = []
        noncontam_reads = []
        log_read_probs = []
        read_probs = []
        all_read_lengths = []
        num_muts = []

        for count, thing in enumerate(samfile):
            if count % 100000 == 0:
                #print(count)
                pass 
            temp_read_freq = [-1]* 30000
            contam_found = False
            temp_read_probs = 0
            temp_log_probs = 1
            read_length = 0
            muts = 0
            pos_found = False
            for (pos, nuc, ref, qual) in zip(thing.get_reference_positions(), \
                thing.query_alignment_sequence, thing.get_reference_sequence(), thing.query_alignment_qualities):
                pos = str(pos+1)
               
                if gt_dict is not None:
                    gt = gt_dict[int(pos)]
                nuc = nuc.upper()
                ref = max(position_depths[pos]["ref"], key=position_depths[pos]["ref"].get)
                if virus_1 is not None:
                    v1_ref = max(virus_1[pos]["ref"], key=virus_1[pos]["ref"].get) #get v1 nucs
                if name == "mix10":
                    v2_ref = max(virus_2[pos]["ref"], key=virus_2[pos]["ref"].get) #get v2 nucs
               
                total_depth = position_depths[pos]['total_depth']
                if total_depth < 10:
                    continue
                
                #this block disgards non informative positions
                """
                pos_alleles = position_depths[pos]['allele']
                pos_alleles = {k: v['count'] / total_depth for k, v in pos_alleles.items()}
                pos_alleles = {k:v for (k,v) in pos_alleles.items() if v > 0.03 and k != ref} 
                if len(pos_alleles) == 0:
                    continue            
                """

                #let's try taking only the pos that don't match the refs
                if (ref != nuc):
                    muts += 1
                else:
                    temp_read_freq[int(pos)] = 0.0
                    continue

                read_length += 1
                allele_depth = position_depths[pos]['allele'][nuc]['count'] 
                freq = allele_depth/total_depth
                temp_read_freq[int(pos)] = freq
     
                if name == 'mix10':
                    #track contam reads
                    if gt == "TRUE" and (ref != nuc) and (ref == v2_ref):
                        contam_found=True
               
                #if int(pos) == 23593:
                #    print(nuc, ref, freq, 'made it')
                #    pos_found = True
                #we care about the actual prob in this freq range
                
                temp_read_probs += freq            
                temp_log_probs += math.log(freq)
           
            dset[count:count+1] = np.array(temp_read_freq)
            dset.resize(count+2, axis=0)
            if contam_found:
                contam_reads.append(count)            
            else:
                noncontam_reads.append(count)
            num_muts.append(muts)
            all_read_lengths.append(read_length)
            if temp_log_probs == 1:
                log_read_probs.append(0)
            else:
                log_read_probs.append(temp_log_probs)
            if read_length == 0:
                read_probs.append(0)
            else:
                #print(temp_read_freq[:100], temp_read_probs/read_length) 
                #sys.exit(0)
                read_probs.append(temp_read_probs/read_length)
            if pos_found:
                print("read prob ", temp_read_probs, read_length, count)
       
        add_info_dict = {"contam_reads":contam_reads, "noncontam_reads":noncontam_reads, \
            "log_read_probs":log_read_probs, "read_probs": read_probs, "read_lengths":all_read_lengths, \
            "num_muts":num_muts}

        with open("/home/chrissy/contam_work/add_info_%s.json" %name, "w") as jfile:
            json.dump(add_info_dict, jfile)    
        
        f.close()
    
    contam_probs = [] 
    noncontam_probs = []

    with open("/home/chrissy/contam_work/add_info_%s.json" %name, "r") as jfile:
        data = json.load(jfile)
    
    log_probs = data['log_read_probs']
    probs = data['read_probs']
    contam_reads = data['contam_reads']
    noncontam_reads = data['noncontam_reads']
    num_muts = data['num_muts']
    read_length = data['read_lengths'] 
  
    """ 
    print('creating visual')
    contam_probs = []
    noncontam_probs = []
    log_contam_probs = []
    log_noncontam_probs = []
    print(len(contam_reads))
    print(len(noncontam_reads))
    print("percent marked com reads: ", len(contam_reads)/(len(contam_reads)+len(noncontam_reads)))
    index_of_possible_contam = []   
  
    x = []
    y = []
    
    for count, value in enumerate(probs):
        if count % 100000 == 0:
            print(count)
        if value == 0:
            continue
        #if count in contam_reads:
        x.append(value)
        #else:
        #    y.append(value)
    
    sns.set_style("darkgrid")
    sns.set(rc = {'figure.figsize':(8,11)})
 
    sns.displot(
    {"all": x, "ignore": y},  # Use a dict to assign labels to each curve
    kind="kde",
    common_norm=False,  # Normalize each distribution independently
    palette=["blue", "red"],  # Use palette for multiple colors
    linewidth=1
    ) 
    plt.xticks(list(np.linspace(0,1,11))) 
    plt.title("comparing %s avg prob of reads for informative pos" %name)
    plt.savefig("/home/chrissy/contam_work/spike_in/images/%s.jpg" %name)
    plt.close()
    """ 
    return
    
 
    with h5py.File("myfile_%s.hdf5" %name, "r") as hfile:
        dset = hfile['default']
        for count, (val,rl) in enumerate(zip(probs, read_length)):
            if count % 100000 == 0:
                print(count)
            if val > threshold and rl > 1:
                index_of_possible_contam.append(count)

    print(len(index_of_possible_contam)) 
    flagged_nucs = {}
    normal_nucs = {}
    pos_depths = position_depths
    for count, thing in enumerate(samfile):
        if count % 100000 == 0:
            print(count)
        for (pos, nuc, ref, qual) in zip(thing.get_reference_positions(), \
            thing.query_alignment_sequence, thing.get_reference_sequence(), thing.query_alignment_qualities):
            pos = str(pos+1)
            nuc = nuc.upper()
            ref = max(position_depths[pos]["ref"], key=position_depths[pos]["ref"].get)
            if nuc == ref:
                continue

            total_depth = position_depths[pos]['total_depth']
            if total_depth < 10:
                continue
               
            allele_depth = position_depths[pos]['allele'][nuc]['count'] 
            freq = allele_depth/total_depth
           
            #this block disgards non informative positions
            pos_alleles = position_depths[pos]['allele']
            pos_alleles = {k: v['count'] / total_depth for k, v in pos_alleles.items()}
            pos_alleles = {k:v for (k,v) in pos_alleles.items() if v > 0.03 and k != ref} 
            if len(pos_alleles) == 0:
                continue                       
            
            letter = nuc

            #average mutation occurs more heavily
            if count in index_of_possible_contam:
                #check if we've added this pos before
                if pos in flagged_nucs:
                    #check if we've seen this nuc before
                    if letter in flagged_nucs[pos]["allele"]:
                        flagged_nucs[pos]['allele'][letter]['count'] += 1
                        flagged_nucs[pos]['allele'][letter]['qual'] += pos_depths[str(pos)]['allele'][letter]['qual']
                        flagged_nucs[pos]['total_depth'] += 1 
                    
                    #else we will add the letter            
                    else:
                        flagged_nucs[pos]['allele'][letter] = {"count":1, \
                            "qual": pos_depths[str(pos)]['allele'][letter]['qual']}
                        flagged_nucs[pos]['total_depth'] += 1
                        
                #else we'll add this position
                else:
                    #print(pos_depths[str(pos)], pos, pos_depths[str(pos-1)])
                    flagged_nucs[pos] = {}
                    flagged_nucs[pos]['allele'] = \
                        {letter:{"count":1, "qual":pos_depths[str(pos)]['allele'][letter]['qual']}}
                    flagged_nucs[pos]['total_depth'] = 1

            #average mutation is rarely seen
            else:
                #check if we've added this pos before
                if pos in normal_nucs:
                    #check if we've seen this nuc before
                    if letter in normal_nucs[pos]["allele"]:
                        normal_nucs[pos]['allele'][letter]['count'] += 1
                        normal_nucs[pos]['allele'][letter]['qual'] += pos_depths[str(pos)]['allele'][letter]['qual']
                        normal_nucs[pos]['total_depth'] += 1 
                    
                    #else we will add the letter            
                    else:
                        normal_nucs[pos]['allele'][letter] = {"count":1, \
                            "qual": pos_depths[str(pos)]['allele'][letter]['qual']}
                        normal_nucs[pos]['total_depth'] += 1
                        
                #else we'll add this position
                else:
                    normal_nucs[pos] = {}
                    normal_nucs[pos]['allele'] = \
                        {letter:{"count":1, "qual":pos_depths[str(pos)]['allele'][letter]['qual']}}
                    normal_nucs[pos]['total_depth'] = 1
 
    return(normal_nucs, flagged_nucs)

def find_contamination(position_depths, flagged_positions, ground_truth, usable_pos, nonflagged_positions, \
    virus_1, virus_2):
    """
    Decides at each position whether a variant is valid or contamination.

    Parameters
    ----------
    position_depths : dict
        Depths at every position.
    flagged_positions : dict 
        Depths for every suspect nuc at every suspect position.

    Returns
    -------
    
    """
    #make ground truth dict for reference   
    zip_iterator = zip(usable_pos, ground_truth) 
    gt_dict = dict(zip_iterator)
    
    #this is where we store what we think this nuc/pos is
    choice = []

    #now we want to iterate every position, and say what we think the nuc is
    for key, value in position_depths.items():
        total_depth = value["total_depth"]
        possible_alleles = value["allele"]
        reference_allele = max(value["ref"], key=value["ref"].get)
        
        m = 0
        r = ''
        for k, v in virus_1[key]['allele'].items():
            if v['count'] > m:
                r = k
                m = v['count']
        v1_ref = r
        m = 0
        r = ''
       
        for k, v in virus_2[key]['allele'].items():
            if v['count'] > m:
                r = k
                m = v['count']
        
        v2_ref = r 
        v1 = virus_1[key]['allele']
        v2 = virus_2[key]['allele']
        
        if total_depth < 10:
            continue
        
        #iterate all indels that make our cuttoff
        for ckey, cvalue in possible_alleles.items():
            try:
                v1_count = v1[ckey]['count']
            except:
                v1_count = 'None'
            try:
                v2_count = v2[ckey]['count']
            except:
                v2_count = "None"
            #matches the reference
            if ckey.lower() == reference_allele.lower():
                continue
            if ckey.lower() == "n":
                continue
            cdepth = cvalue['count']
            cqual = cvalue['qual']/cvalue['count']
            cvalue = cvalue['count']/total_depth
            #we think this is a normal variant
            if cvalue <= 0.03:
                temp_dict = {"pos":int(key), "ref": reference_allele, "allele":ckey, \
                    "status": "low frequency", "freq":cvalue, 'avg_qual':cqual, \
                    "contam_freq":"NA", "total_depth":total_depth, "count": cdepth, "v1_ref": v1_ref, \
                    "v2_ref": v2_ref, "v1_count":v1_count, "v2_count":v2_count}
                choice.append(temp_dict)
                
            #it doesn't match the reference but it's not low enough to be flagged as contamination
            else:
                #we go look for it's occurence in the pos in contaminated samples
                if str(key) in flagged_positions: 
                    #did we see this nuc in this pos in the flagged reads?
                    if ckey not in flagged_positions[str(key)]['allele']:
                        print("Didn't occur in contamination!")
                        temp_dict = {"pos":int(key), "ref":reference_allele,"allele":ckey, \
                            "status": "normal base", "freq":cvalue, 'avg_qual':cqual, \
                            "contam_freq":"NA", "total_depth":total_depth, "count":cdepth, "v1_ref":v1_ref, \
                            "v2_ref":v2_ref,"v1_count":v1_count, "v2_count":v2_count}
                        choice.append(temp_dict) 
                    
                    #we did see this in the contamination
                    else:
                        #total contamination pos depth
                        contam_depth = flagged_positions[str(key)]['total_depth']
                        temp_allele = flagged_positions[str(key)]['allele']
                        #freq occuring in contam
                        temp_normed_allele = temp_allele[ckey]['count'] / cdepth
                        #we also go look for it in the noncont to compare counts
                        try:
                            noncontam_counts = nonflagged_positions[str(key)]['allele'][ckey]['count']
                        except:
                            temp_dict = {"pos":int(key), "ref": reference_allele, "allele":ckey, \
                                "status": "contaminant", "freq":cvalue, 'avg_qual':cqual, \
                                "contam_freq":temp_normed_allele, "total_depth":total_depth, \
                                "count":cdepth, "v1_ref":v1_ref, "v2_ref":v2_ref,"v1_count":v1_count, "v2_count":v2_count}
                            choice.append(temp_dict)
                            continue
                        
                        #we divide it up into ranges based on how certain we are it's a contaminant
                        if temp_normed_allele > 0.5:
                            temp_dict = {"pos":int(key), "ref": reference_allele, "allele":ckey, \
                                "status": "contaminant", "freq":cvalue, 'avg_qual':cqual, \
                                "contam_freq":temp_normed_allele, "total_depth":total_depth, \
                                "count":cdepth, "v2_ref":v2_ref, "v1_ref":v1_ref, \
                                "v1_count":v1_count, "v2_count":v2_count}
                            choice.append(temp_dict)
                            continue
                        
                        #occurs in lower percentage of contaminants
                        elif (temp_normed_allele > 0.10):
                            if cqual <= 1:
                            #let's also look at the quality
                                temp_dict = {"pos":int(key), "ref": reference_allele, "allele":ckey, \
                                    "status": "contaminant", "freq":cvalue, 'avg_qual':cqual, \
                                    "contam_freq":temp_normed_allele, "total_depth":total_depth, \
                                    "count":cdepth, "v1_ref":v1_ref, "v2_ref":v2_ref, \
                                    "v1_count":v1_count, "v2_count":v2_count}
                                choice.append(temp_dict)
                                continue
                            else:
                                temp_dict = {"pos":int(key), "ref": reference_allele, "allele":ckey, \
                                    "status": "normal base", "freq":cvalue, 'avg_qual':cqual, \
                                    "contam_freq":temp_normed_allele, "total_depth":total_depth, \
                                    "count":cdepth, "v1_ref":v1_ref, "v2_ref":v2_ref, \
                                    "v1_count":v1_count, "v2_count":v2_count}
                                choice.append(temp_dict)
                                continue
                                              
                #we didn't find any of this position in flagged reads 
                else:
                    print("Not in flagged read", key)
                    temp_dict = {"pos":int(key), "ref":reference_allele,"allele":ckey, \
                        "status": "normal base", "freq":cvalue, 'avg_qual':cqual,
                        "contam_freq":"NA", "total_depth":total_depth, "count":cdepth, "v1_ref": v1_ref, \
                        "v2_ref":v2_ref, "v1_count":v1_count, "v2_count":v2_count}
                    choice.append(temp_dict)
              
    return(choice)
     
def manipulate_spreadsheet(filename, ground_truth_filename):
    df = pd.read_csv(filename)
    ground_truth_df = pd.read_csv(ground_truth_filename)
    
    current_pos = ground_truth_df['Position'].tolist()
    print(current_pos[0], current_pos[-1]) 
    merge_df = df.merge(ground_truth_df, right_on="Position", left_on="pos", how='right')
    #merge_df = merge_df[merge_df['True/False/Remove'] != "Remove"]
   
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0
    unclear = 0
    no_pred = 0
    new_row = []
    for index,row in merge_df.iterrows():
        #print(row)
        prediction = row["status"]
        gt = row["True/False/Remove"]
        nuc = row['allele']
        freq = row['freq']
        v1 = row['v1_ref']
        v2 = row['v2_ref']
        if (prediction == "normal base" or prediction == "low frequency") and gt == 'Remove':
            if prediction == "normal base" and nuc == v2:
                false_neg += 1
                new_row.append("false neg")
            else:
                true_neg += 1
                new_row.append("true neg")
        elif (prediction == "contaminant") and gt == "Remove":
            if nuc == v2:
                true_pos += 1
                new_row.append("true pos")
            else:
                false_pos += 1
                new_row.append("false pos")
        elif (prediction == "normal base" or prediction == "low frequency") and gt == "FALSE":
            true_neg += 1
            new_row.append("true neg")
        elif prediction == "contaminant" and gt == "TRUE":
            #condition relevant to the dataset 
            if freq > 0.03 or freq <= 0.15:
                true_pos += 1
                new_row.append("true pos")
            #it's the wrong base, false pos
            else:
                new_row.append("false pos")
                false_pos += 1
        elif (prediction == "normal base" or prediction == "low frequency") and gt == "TRUE":
            #this is our known contam range
            if freq < 0.03 or freq >= 0.15:
                new_row.append("true neg")
                true_neg += 1
            else:
                new_row.append("false neg")
                false_neg += 1
        elif prediction == "contaminant" and gt == "FALSE":
            new_row.append("false pos")
            false_pos += 1
        else:
            #print("MISTAKE", prediction, gt)
            no_pred += 1
            new_row.append("No prediction")
        
    print("True Pos: ", true_pos)
    print("True Neg: ", true_neg)
    print("False Pos: ", false_pos)
    print("False Neg: ", false_neg)
    print("Unclear: ", unclear)
    print("No Prediction: ", no_pred)
    merge_df['TFP'] = new_row
    merge_df.to_csv("test.csv")

def look_for_viable_pos(ground_truth_filename):
    df = pd.read_csv(ground_truth_filename)
    #df = ground_truth_df[ground_truth_df['True/False/Remove'] != "Remove"]
    return(df["Position"].tolist(), df['True/False/Remove'].tolist())

def main():
    """
	Proof of concept code for identifying contaminants.
	"""
    #code options
    calc_pos_depths=False
    flag_reads=True
    output_ground_truth=True
    create_new_hdf5=True
    
    #files and names
    name = "mix10"
    filename = "./test.csv"
    target = "exp" #exp, virus_1, or virus_2
    threshold = 0.05
    ground_truth_filename = "./ZIKV-intrahost_True-False-Remove.csv"
   
    usable_pos,ground_truth = look_for_viable_pos(ground_truth_filename)
    zip_iterator = zip(usable_pos, ground_truth) 
    gt_dict = dict(zip_iterator) 

    if target == 'exp':
        bam = "./bam/test.sorted.calmd.bam"
    elif target == 'virus_1':
        bam = "./bam/metagenomics_virus1.sorted.bam"
    elif target == "virus_2":
        bam = './bam/metagenomics_virus2.sorted.bam'
 
    if calc_pos_depths: 
        position_depths = calculate_positional_depths(bam) 
        with open('pos_depths_%s.json' %name ,'w') as jfile:
            json.dump(position_depths, jfile)
    else: 
        with open('pos_depths_%s.json' %name, 'r') as jfile:
            position_depths = json.load(jfile)
        with open('pos_depths_virus_1.json', 'r') as jfile:
            virus_1 = json.load(jfile)
        with open('pos_depths_virus_2.json', 'r') as jfile:
            virus_2  = json.load(jfile)
    
    if flag_reads:
        normal_nucs, flagged_nucs  = calculate_read_probability(position_depths, bam, gt_dict, virus_1, \
            virus_2, name, threshold, create_new_hdf5)
        dump_dict = {"normal_nucs":normal_nucs, "flagged_nucs":flagged_nucs}
        
        with open("flagged_depths_%s.json" %name, "w") as jfile:
            json.dump(dump_dict, jfile)
        nonflagged_positions = normal_nucs
        flagged_positions = flagged_nucs
    else: 
        with open("flagged_depths_%s.json" %name, "r") as jfile:
            dump_dict = json.load(jfile)
        nonflagged_positions = dump_dict["normal_nucs"]
        flagged_positions = dump_dict["flagged_nucs"]
    
    #go through and try and use frequency to determine contaminant, using flagged reads
    temp_list = find_contamination(position_depths, flagged_positions, ground_truth, usable_pos, \
        nonflagged_positions, virus_1, virus_2)
    
    df = pd.DataFrame(temp_list)
    df.to_csv(filename)

    if output_ground_truth:
        #actually calculate T/F P/N
        manipulate_spreadsheet(filename, ground_truth_filename)
        

if __name__ == "__main__":
    main()
