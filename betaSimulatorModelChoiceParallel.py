import msprime
import numpy as np
import matplotlib.pyplot as plt
import math
import tskit
import scipy
import allel
import pandas as pd
import multiprocessing



r_chrom = 1e-8 #Recombination rate
r_break = math.log(2) #Recombination rate needed to satisfy probability 2^-t inheritance of two chromsomes
chrom_positions = [0, 1e6, 2e6, 3e6] #1Mb chromosome sizes
map_positions = [
    chrom_positions[0],
    chrom_positions[1],
    chrom_positions[1] + 1,
    chrom_positions[2],
    chrom_positions[2] + 1,
    chrom_positions[3]
]
rates = [r_chrom, r_break, r_chrom, r_break, r_chrom] 
rate_map = msprime.RateMap(position=map_positions, rate=rates) #Rate map for separate chromosomes


def task(alpha, reps):
    data = [] #initialize list to store summary statistics 
    sample_size = 38
    Ne = 1e5
    for i in range(reps):
        ts = msprime.sim_ancestry( #constant population model with beta coalescent
            samples = sample_size,
            population_size = Ne,
            recombination_rate = rate_map,
            model=msprime.BetaCoalescent(alpha = alpha),
            #random_seed=1234,
        )
        

        mts = msprime.sim_mutations(ts, rate=1e-8, random_seed=5678) #simulate mutations on treekit

        np.set_printoptions(legacy="1.21") #exclude dbtype from np arrays
        summary_statistics = [] #Initialize list of summary statistics
        
        if alpha == 1.9:
            summary_statistics.append(1) #First column corresponds to model index
        elif alpha == 1.7:
            summary_statistics.append(2)
        elif alpha == 1.5:
            summary_statistics.append(3)
        elif alpha == 1.3:
            summary_statistics.append(4)
        elif alpha == 1.1:
            summary_statistics.append(5)
        summary_statistics.append(Ne) #Second column is Ne
        summary_statistics.append(alpha) #Third column is alpha parameter
        summary_statistics.append(1) #Fourth column is rho/theta
        S = mts.get_num_mutations()
        summary_statistics.append(S) #Fifth column is number of segregating sites
        normalized_S = mts.segregating_sites(span_normalise=True)
        summary_statistics.append(normalized_S) #Sixth column is span normalized S
        pi = mts.diversity()
        summary_statistics.append(pi) #Seventh column is nucleotide diversity
        

        afs = mts.allele_frequency_spectrum(span_normalise=False, polarised=False)

        afs_entries = [] #initialize list of afs entries. Each element corresponds to an allele's frequency as a proportion e.g. singleton, doubleton, etc.

        for x in range(1, sample_size + 2):
            num_mutations = afs[x]
            l = [x/(sample_size*2)] * int(num_mutations) #create list of allele frequency with length of number of mutations
            afs_entries.extend(l) #extend afs_entries list by elements of the new list
        afs_entries = np.array(afs_entries) 
        #len(afs_entries)

        afs_quant = np.quantile(afs_entries, [0.1, 0.3, 0.5, 0.7, 0.9])
        summary_statistics.append(afs_quant[0]) #8th column is AFS quantile 0.1
        summary_statistics.append(afs_quant[1]) #9th column 0.3
        summary_statistics.append(afs_quant[2]) #10th column 0.5
        summary_statistics.append(afs_quant[3]) #11th column 0.7
        summary_statistics.append(afs_quant[4]) #12th column 0.9
        


        num_windows = 30
        D_array = mts.Tajimas_D(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
        summary_statistics.append(np.nanmean(D_array)) #13th column is mean Tajima's D
        summary_statistics.append(np.nanvar(D_array)) #14th column is variance of Tajima's D
        


        ts_chroms = []
        for j in range(len(chrom_positions) - 1): #split genome into chromosomes
            start, end = chrom_positions[j: j + 2]
            chrom_ts = mts.keep_intervals([[start, end]], simplify=False).trim()
            ts_chroms.append(chrom_ts)
            #print(chrom_ts.sequence_length)


        gn = mts.genotype_matrix()
        # print("Converted to genotype matrix...")
        r = allel.rogers_huff_r(gn)
        # print("Calculated r...")
        s = scipy.spatial.distance.squareform(r ** 2) #calculate r^2

        arr = mts.sites_position #array of site positions
        result = abs(arr[:, None] - arr) #broadcast subtraction to create matrix of pairwise distances between sites

        #Get LD only on same chromosomes
        chrom1_mut_num = ts_chroms[0].get_num_mutations()
        chrom1_ld = s[:chrom1_mut_num,:chrom1_mut_num]

        chrom2_mut_num = ts_chroms[1].get_num_mutations()
        chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
        chrom2_ld = s[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

        chrom3_mut_num = ts_chroms[2].get_num_mutations()
        total_mut_num = chrom1and2_mut_num + chrom3_mut_num
        chrom3_ld = s[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]


        #Upper triangle of matrix to get rid of duplicated values
        chrom1_ld = chrom1_ld[np.triu_indices_from(chrom1_ld)]
        chrom2_ld = chrom2_ld[np.triu_indices_from(chrom2_ld)]
        chrom3_ld = chrom3_ld[np.triu_indices_from(chrom3_ld)]



        r2 = np.concatenate((chrom1_ld, chrom2_ld, chrom3_ld))
        r2_quant = np.nanquantile(r2, [0.1,0.3,0.5,0.7,0.9])
        #r2_quant


        summary_statistics.append(r2_quant[0]) #15th-21st columns are r^2 quantiles, mean, and variance
        summary_statistics.append(r2_quant[1])
        summary_statistics.append(r2_quant[2])
        summary_statistics.append(r2_quant[3])
        summary_statistics.append(r2_quant[4])
        summary_statistics.append(np.nanmean(r2))
        summary_statistics.append(np.nanvar(r2))
        
        scaled_ld = result * s #scale LD by distance between pairs of SNPs; matrix multiplication of distances times r^2

        #Same procedure as r^2 to get only LD on same chromosome
        chrom1_mut_num = ts_chroms[0].get_num_mutations()
        chrom1_scaled_ld = scaled_ld[:chrom1_mut_num,:chrom1_mut_num]

        chrom2_mut_num = ts_chroms[1].get_num_mutations()
        chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
        chrom2_scaled_ld = scaled_ld[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

        chrom3_mut_num = ts_chroms[2].get_num_mutations()
        total_mut_num = chrom1and2_mut_num + chrom3_mut_num
        chrom3_scaled_ld = scaled_ld[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]



        chrom1_scaled_ld = chrom1_scaled_ld[np.triu_indices_from(chrom1_scaled_ld)]
        chrom2_scaled_ld = chrom2_scaled_ld[np.triu_indices_from(chrom2_scaled_ld)]
        chrom3_scaled_ld = chrom3_scaled_ld[np.triu_indices_from(chrom3_scaled_ld)]



        scaled_r2 = np.concatenate((chrom1_scaled_ld, chrom2_scaled_ld, chrom3_scaled_ld))
        scaled_r2_quant = np.nanquantile(scaled_r2, [0.1,0.3,0.5,0.7,0.9])
        scaled_r2_quant


        summary_statistics.append(scaled_r2_quant[0]) #22nd-28th columns scaled r^2
        summary_statistics.append(scaled_r2_quant[1])
        summary_statistics.append(scaled_r2_quant[2])
        summary_statistics.append(scaled_r2_quant[3])
        summary_statistics.append(scaled_r2_quant[4])
        summary_statistics.append(np.nanmean(scaled_r2))
        summary_statistics.append(np.nanvar(scaled_r2))

        #Interchromosomal LD
        chrom1_ild = s[:chrom1_mut_num,chrom1_mut_num:]
        chrom1_ild = np.matrix.flatten(chrom1_ild)
        chrom2_ild_a = s[chrom1_mut_num:chrom1and2_mut_num,chrom1and2_mut_num:]
        chrom2_ild_b = s[chrom1_mut_num:chrom1and2_mut_num,chrom1_mut_num:chrom1and2_mut_num]
        chrom2_ild_a = np.matrix.flatten(chrom2_ild_a)
        chrom2_ild_b = np.matrix.flatten(chrom2_ild_b)
        chrom2_ild = np.concatenate((chrom2_ild_a, chrom2_ild_b))
        chrom3_ild = s[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:chrom3_mut_num]
        chrom3_ild = np.matrix.flatten(chrom3_ild)
        #np.nanmean(chrom2_ild)


        ild_all = np.concatenate((chrom1_ild, chrom2_ild, chrom3_ild))
        ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9])


        summary_statistics.append(ild_quant[0]) #29th-35th columns ILD
        summary_statistics.append(ild_quant[1])
        summary_statistics.append(ild_quant[2])
        summary_statistics.append(ild_quant[3])
        summary_statistics.append(ild_quant[4])
        summary_statistics.append(np.nanmean(ild_all))
        summary_statistics.append(np.nanvar(ild_all))

        #Append to data list to form data frame outside loop (faster than appending data frame)
        data.append(summary_statistics)
    return data

if __name__ == '__main__':
    alpha_list = [1.9, 1.7, 1.5, 1.3, 1.1]  # List of values to process
    reps =   # How many times you want to process each value

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=5) as pool:
        # Use starmap to apply the function with multiple arguments (value and repetitions)
        results = pool.starmap(task, [(alpha, reps) for alpha in alpha_list])
        # Flatten the list since results will be a list of lists
        flattened_results = [item for sublist in results for item in sublist]
        summary_list = pd.DataFrame(flattened_results)

        summary_list.to_csv('summary_statistics_parallel.csv', index = False, mode = 'a')




