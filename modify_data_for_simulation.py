import os
import sys
import pysam
import numpy as np

from joblib import Parallel, delayed

sys.path.insert(1, '../contamination_work')
from amplicon_variance import parse_key_mutations

def myround(x, base=5):
    return base * round(x/base)

def mix_simulated_data():
    strain_list = ['alpha', 'gamma', 'beta', 'delta']
    strain_list = ['alpha','gamma']
    percent_list = [round(x,2) for x in list(np.arange(0.05,1.05,0.05))]
    current_simulated_reads = os.listdir('./save_simulated')

    #loop through all combinations of data    
    for strain in strain_list:
        for strain_2 in strain_list:
            if strain == strain_2:
                continue
            for percent_1 in percent_list:
                percent_1_str = str(myround(int(percent_1*100)))
                if percent_1_str == '5':
                    percent_1_str = '05'
                percent_2 = 1 - percent_1 
                
                percent_2_str = str(myround(int(percent_2*100)))
                
                if percent_2_str == '5':
                    percent_2_str = '05'
                merge_file_1 = './save_simulated/simulated_%s_%s.bam' %(strain, percent_1_str)
                merge_file_2 = './save_simulated/simulated_%s_%s.bam' %(strain_2, percent_2_str)
                final_filename = './final_simulated_data/simulated_%s_%s_%s_%s.bam' \
                   %(strain, strain_2, percent_1_str, percent_2_str)            
                alt_filename = './final_simulated_data/simulated_%s_%s_%s_%s.bam' \
                    %(strain_2, strain, percent_2_str, percent_1_str)
                if os.path.isfile(final_filename) or os.path.isfile(alt_filename):
                    continue    
                cmd0 = 'cp %s ./bam_merge' %merge_file_1
                cmd1 = 'cp %s ./bam_merge' %merge_file_2
                cmd2 = 'samtools merge %s ./bam_merge/*.bam' %final_filename
                cmd3 = 'rm ./bam_merge/*'
                os.system(cmd0)
                os.system(cmd1)
                os.system(cmd2)
                os.system(cmd3)

def generate_simulated_reads():
    percent_list = [round(x,2) for x in list(np.arange(0.05,1.05,0.05))]
    strain_list = ['alpha', 'beta', 'gamma', 'delta']
    strain_list = ['alpha', 'gamma']
    #percent_list = [0.30, 0.70]
    
    n_jobs = -35
    for strain in strain_list:
        results = Parallel(n_jobs=n_jobs)(delayed(parallel_generate_strains)(percent, strain) \
            for percent in percent_list)
        

def parallel_generate_strains(percent, strain):
    vcf_filename = './simulated_vcf/%s_mod.vcf' %strain
    basename = "%s_%s" %(strain, percent)
    percent_str = str(myround(int(percent*100)))

    if percent_str == '5':
        percent_str = '05'

    output_filename = './save_simulated/simulated_%s_%s.bam' %(strain, percent_str)
    if os.path.isfile(output_filename):
        return(0)
    else:
        output_1 = "%s_1.fq" %(basename)
        output_2 = "%s_2.fq" %(basename)
        num_reads = int((100000*percent)/2)
        print(num_reads)
        cmd = 'reseq illuminaPE -j 32 -r ../sequence.fasta -b aaron.preprocessed.bam -V %s -1 %s -2 %s --numReads %s -v %s --noBias' %(vcf_filename, output_1, output_2, str(num_reads), vcf_filename)
        cmd2 = 'bwa mem -t 32 ../sequence.fasta %s %s | samtools view -b -F 4 -F 2048 | samtools sort -o %s' %(output_1, output_2, output_filename)
        os.system(cmd)
        os.system(cmd2)
        os.system("rm %s" %output_1)
        os.system("rm %s" %output_2)
    return(0)

def modify_vcf():
    """
    Remove unexpected variants from the vcf file based on outbreak.info
    information about key mutations.
    """
    
    strain_list = ['alpha', 'gamma', 'beta', 'delta']
    percent_list = [round(x,2) for x in list(np.arange(0.05,1,0.05))]
    percent_list = [1]
    total_pos_dict = parse_key_mutations() 
    
    for strain_1 in strain_list:
        vcf_1 = './%s.vcf' %strain_1
        mutations_1 = total_pos_dict[strain_1]
        
        for strain_2 in strain_list:
            if strain_1 == strain_2:
                continue
            mutations_2 = total_pos_dict[strain_2]    
            vcf_2 = './%s.vcf' %strain_2
            for percent_1 in percent_list: 
                percent_1_str = str(myround(int(percent_1*100)))
                if percent_1_str == '5':
                    percent_1_str = '05'
                percent_2 = 1 - percent_1 
                percent_2_str = str(myround(int(percent_2*100)))
                
                if percent_2_str == '5':
                    percent_2_str = '05'
                final_filename = './simulated_vcf/%s_%s_%s_%s.vcf' %(strain_1, strain_2, percent_1_str, percent_2_str)
 
                lines_1 = get_all_lines(vcf_1, mutations_1, percent_1)
                lines_2 = get_all_lines(vcf_2, mutations_2, percent_2, header=False)
                lines_1.extend(lines_2)
                ordered_lines = sort_vcf_lines(lines_1)
                with open(final_filename, 'w') as wfile:
                    for line in ordered_lines:
                        wfile.write(line + "\n")
def mod_single_vcf():
    strain_list = ['alpha', 'gamma', 'beta', 'delta']
    percent_list = [1]
    total_pos_dict = parse_key_mutations() 
    
    for strain_1 in strain_list:
        vcf_1 = './%s.vcf' %strain_1
        mutations_1 = total_pos_dict[strain_1]
        
        for percent_1 in percent_list:       
            final_filename = './simulated_vcf/%s_mod.vcf' %(strain_1)

            lines_1 = get_all_lines(vcf_1, mutations_1, percent_1)
            with open(final_filename, 'w') as wfile:
                for line in lines_1:
                    wfile.write(line + "\n")

def sort_vcf_lines(lines):
    """
    Helper function to put vcf lines in a particular order.
    """
    reordered_lines = []
    positions = []
    index = []
    for count, line in enumerate(lines):
        if line.startswith("#"):
            reordered_lines.append(line)
            continue
        positions.append(int(line.split("\t")[1]))
        index.append(count)

    #take care of shared mutations in paired files
    u_pos, u_pos_count = np.unique(positions, return_counts=True)
    double_pos = [p for p,c in zip(u_pos, u_pos_count) if c > 1]
    seen_pos = []
    new_list = [x for _, x in sorted(zip(positions, index), key=lambda pair: pair[0])]
    for item in new_list:
        for count, line in enumerate(lines):
            if count == item:
                line_list = line.split("\t")
                if line_list[1] in seen_pos:
                    continue
                if int(line_list[1]) in double_pos:
                    line = reset_line_100(line_list)
                    seen_pos.append(line_list[1])
                reordered_lines.append(line)
    return(reordered_lines)

def reset_line_100(line_list):
    info = line_list[7].split(';')
    #info[1] = "AB=1.0"
    alt = info[-2].split(',')
    alt[-1] = '500'
    alt[-2] = '500'
    alt[1] = '0'
    alt[0] = 'DP4=0'
    info[-2] = ','.join(alt)
    line_list[7] = ';'.join(info)
    line_mod = '\t'.join(line_list)
    
    return(line_mod)

def reseq_wrapper():
    all_vcf = [os.path.join('./simulated_vcf',x) for x in os.listdir('./simulated_vcf')]
    for vcf in all_vcf:
        parallel_generate_strains(vcf)
        sys.exit(0)
def remove_reads():
    samfile = pysam.AlignmentFile("aaron_320.bam", "rb")

    all_read_names = []
    for read in samfile:
        all_read_names.append(read.query_name)

    reads, counts = np.unique(all_read_names, return_counts=True)
    remove = []
    for r, c in zip(reads,counts):
        if c < 2:
            remove.append(r)
    samfile.close()
    print(len(remove))

    samfile_2 = pysam.AlignmentFile("aaron_320.bam", "rb")
    outfile = pysam.AlignmentFile("aaron.preprocessed.bam", "wb", template=samfile_2)
    for s in samfile_2:
        if s.query_name in remove:
            continue
        else:
            outfile.write(s)

def get_all_lines(vcf_file, mutations_keep, percent, header=True):
    """
    Helper function to get the applicables lines from the vcf file.
    """
    write_lines = []
    alt_total = 1000 * percent
    ref_total = 1000 * (1-percent)
    alt_half = int(alt_total / 2)
    ref_half = int(ref_total / 2)
    with open(vcf_file, 'r') as vfile:
        for line in vfile:
            line = line.strip()
            if line.startswith('#'):
                if header:
                    write_lines.append(line)
                    #if line == '##INFO=<ID=DP,Number=1,Type=Integer,Description="Raw read depth">':
                    #write_lines.append('##INFO=<ID=AD,Number=1,Type=Integer,Description="Depth of Alt allele">')
            else:
                line_list = line.split("\t")                
                info = line_list[7].split(';')
                #change total reads to be 1000
                info[0] = "DP=1000"             
                alt = info[-2].split(',')
                alt[-1] = str(alt_half)
                alt[-2] = str(alt_half)
                alt[1] = str(ref_half) 
                alt[0] = alt[0].split('=')[0] + '=' + str(ref_half)
                info[-2] = ','.join(alt)
                #info.insert(1,"AB=%s" %str(percent))
                line_list[7] = ';'.join(info)
                line_mod = '\t'.join(line_list)
                pos = line_list[1]
                if int(pos) not in mutations_keep:
                    continue
                else:
                    write_lines.append(line_mod)

    return(write_lines)

if __name__ == "__main__":
    #parallel_generate_strains('./simulated_vcf/beta_mod.vcf')
    #remove_reads()
    #reseq_wrapper()
    #mod_single_vcf()
    #modify_vcf()
    #generate_simulated_reads()
    mix_simulated_data()
