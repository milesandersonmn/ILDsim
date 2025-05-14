import msprime
import numpy as np
import math
import tskit
import scipy
import allel
from pandas import DataFrame


def alpha1_9(arg):
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


    #del r_chrom, r_break, map_positions, rates

    #data = [] #initialize list to store summary statistics 
    alpha = 1.9
    sample_size = 50
    Ne = np.random.randint(25000, 100000)
    #Ne = 1e5
    #for i in range(reps):
    ts = msprime.sim_ancestry( #constant population model with beta coalescent
        samples = sample_size,
        population_size = Ne,
        recombination_rate = rate_map,
        model=msprime.BetaCoalescent(alpha = alpha),
        #random_seed=1234,
    )

    #del rate_map

    mts = msprime.sim_mutations(ts, rate=1e-8) #simulate mutations on treekit

    

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
    S = mts.num_sites
    summary_statistics.append(S) #Fifth column is number of segregating sites
    normalized_S = mts.segregating_sites(span_normalise=True)
    summary_statistics.append(normalized_S) #Sixth column is span normalized S
    
    num_windows = 30
    pi_array = mts.diversity(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
    summary_statistics.append(np.nanmean(pi_array))
    summary_statistics.append(scipy.stats.hmean(pi_array, nan_policy = 'omit')) #Seventh column is mean pi
    summary_statistics.append(np.nanvar(pi_array)) #Eighth column is variance of pi
    summary_statistics.append(np.nanstd(pi_array))
    pi = mts.diversity()
    summary_statistics.append(pi) #Ninth column is nucleotide diversity


    afs = mts.allele_frequency_spectrum(span_normalise=False, polarised=False)

    afs_entries = [] #initialize list of afs entries. Each element corresponds to an allele's frequency as a proportion e.g. singleton, doubleton, etc.

    for x in range(1, sample_size + 2):
        num_mutations = afs[x]
        l = [x/(sample_size*2)] * int(num_mutations) #create list of allele frequency with length of number of mutations
        afs_entries.extend(l) #extend afs_entries list by elements of the new list
    afs_entries = np.array(afs_entries) 
    #len(afs_entries)
    
    summary_statistics.append(afs[1]/sum(afs))
    summary_statistics.append(afs[2]/sum(afs))
    summary_statistics.append(afs[3]/sum(afs))
    summary_statistics.append(afs[4]/sum(afs))
    summary_statistics.append(afs[5]/sum(afs))
    summary_statistics.append(afs[6]/sum(afs))
    summary_statistics.append(afs[7]/sum(afs))
    summary_statistics.append(afs[8]/sum(afs))
    summary_statistics.append(afs[9]/sum(afs))
    summary_statistics.append(afs[10]/sum(afs))
    summary_statistics.append(afs[11]/sum(afs))
    summary_statistics.append(afs[12]/sum(afs))
    summary_statistics.append(afs[13]/sum(afs))
    summary_statistics.append(afs[14]/sum(afs))
    summary_statistics.append(sum(afs[15:])/sum(afs))

    afs_quant = np.quantile(afs_entries, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(afs_quant[0]) #10th column is AFS quantile 0.1
    summary_statistics.append(afs_quant[1]) #11th column 0.3
    summary_statistics.append(afs_quant[2]) #12th column 0.5
    summary_statistics.append(afs_quant[3]) #13th column 0.7
    summary_statistics.append(afs_quant[4]) #14th column 0.9

    del afs_entries
    del afs_quant

    D_array = mts.Tajimas_D(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
    summary_statistics.append(np.nanmean(D_array)) #15th column is mean Tajima's D
    summary_statistics.append(np.nanvar(D_array)) #16th column is variance of Tajima's D
    summary_statistics.append(np.nanstd(D_array))
    del D_array

    ts_chroms = []
    for j in range(len(chrom_positions) - 1): #split genome into chromosomes
        start, end = chrom_positions[j: j + 2]
        chrom_ts = mts.keep_intervals([[start, end]], simplify=False).trim()
        ts_chroms.append(chrom_ts)
        #print(chrom_ts.sequence_length)

    

    gn = mts.genotype_matrix()
    # print("Converted to genotype matrix...")
    r = allel.rogers_huff_r(gn)

    del gn
    # print("Calculated r...")
    s = scipy.spatial.distance.squareform(r ** 2) #calculate r^2

    arr = mts.sites_position #array of site positions
    pairwise_distances = abs(arr[:, None] - arr) #broadcast subtraction to create matrix of pairwise distances between sites

    h = allel.HaplotypeArray(mts.genotype_matrix())
    #allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))
    summary_statistics.append(np.nanmean(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    summary_statistics.append(np.nanstd(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    summary_statistics.append(np.nanvar(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    hamming_distances = scipy.spatial.distance.pdist(h, metric='hamming') * h.shape[1]
    pairwise_matrix = scipy.spatial.distance.squareform(hamming_distances)
    hamming_array = pairwise_matrix[np.triu_indices_from(pairwise_matrix, k=1)]
    hamming_quant = np.quantile(hamming_array, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(hamming_quant[0]) #27th column is AFS quantile 0.1
    summary_statistics.append(hamming_quant[1]) #28th column 0.3
    summary_statistics.append(hamming_quant[2]) #29th column 0.5
    summary_statistics.append(hamming_quant[3]) #30th column 0.7
    summary_statistics.append(hamming_quant[4]) #31 column 0.9
    summary_statistics.append(np.nanmean(hamming_array))
    summary_statistics.append(np.nanstd(hamming_array))
    summary_statistics.append(np.nanvar(hamming_array))
    #scaled_ld = np.multiply(result, s) #scale LD by distance between pairs of SNPs; matrix multiplication of distances times r^2

    del arr, ts
    #Get the lengths of homozygous runs
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_homozygous = pairwise_distances[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_homozygous = pairwise_distances[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_homozygous = pairwise_distances[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]

    

    chrom1_homozygous = chrom1_homozygous[np.triu_indices_from(chrom1_homozygous, k=1)]
    chrom2_homozygous = chrom2_homozygous[np.triu_indices_from(chrom2_homozygous, k=1)]
    chrom3_homozygous = chrom3_homozygous[np.triu_indices_from(chrom3_homozygous, k=1)]



    homozygous = np.concatenate((chrom1_homozygous, chrom2_homozygous, chrom3_homozygous))

    del chrom1_homozygous, chrom2_homozygous, chrom3_homozygous

    homozygous_quant = np.nanquantile(homozygous, [0.1,0.3,0.5,0.7,0.9])


    summary_statistics.append(homozygous_quant[0]) #17th-23rd columns lengths of homozygosity
    summary_statistics.append(homozygous_quant[1])
    summary_statistics.append(homozygous_quant[2])
    summary_statistics.append(homozygous_quant[3])
    summary_statistics.append(homozygous_quant[4])
    summary_statistics.append(np.nanmean(homozygous))
    summary_statistics.append(scipy.stats.hmean(homozygous, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(homozygous))
    summary_statistics.append(np.nanstd(homozygous))

    del homozygous, homozygous_quant, pairwise_distances
    
    #Get LD only on same chromosomes
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_ld = s[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_ld = s[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_ld = s[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]

    
    #Upper triangle of matrix to get rid of duplicated values
    chrom1_ld = chrom1_ld[np.triu_indices_from(chrom1_ld, k=1)]
    chrom2_ld = chrom2_ld[np.triu_indices_from(chrom2_ld, k=1)]
    chrom3_ld = chrom3_ld[np.triu_indices_from(chrom3_ld, k=1)]



    r2 = np.concatenate((chrom1_ld, chrom2_ld, chrom3_ld))
    r2_quant = np.nanquantile(r2, [0.1,0.3,0.5,0.7,0.9])
    
    del chrom1_ld, chrom2_ld, chrom3_ld


    summary_statistics.append(r2_quant[0]) #24th-30th columns are r^2 quantiles, mean, and variance
    summary_statistics.append(r2_quant[1])
    summary_statistics.append(r2_quant[2])
    summary_statistics.append(r2_quant[3])
    summary_statistics.append(r2_quant[4])
    summary_statistics.append(np.nanmean(r2))
    summary_statistics.append(scipy.stats.hmean(r2, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(r2))
    summary_statistics.append(np.nanstd(r2))

    del r2, r2_quant

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

    del chrom1_mut_num, chrom2_mut_num, chrom3_mut_num, chrom1and2_mut_num, total_mut_num
    ild_all = np.concatenate((chrom1_ild, chrom2_ild, chrom3_ild))
    ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9])

    del chrom1_ild, chrom2_ild, chrom3_ild

    summary_statistics.append(ild_quant[0]) #31st-37th columns ILD
    summary_statistics.append(ild_quant[1])
    summary_statistics.append(ild_quant[2])
    summary_statistics.append(ild_quant[3])
    summary_statistics.append(ild_quant[4])
    summary_statistics.append(np.nanmean(ild_all))
    summary_statistics.append(scipy.stats.hmean(ild_all, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(ild_all))
    summary_statistics.append(np.nanstd(ild_all))

    del ild_all, ild_quant
    #Append to data list to form data frame outside loop (faster than appending data frame)
    #data.append(summary_statistics)
    x = DataFrame(summary_statistics).T

    x.to_csv('summary_statistics_improved.csv', index = False, mode = 'a', header = False)
    

def alpha1_7(arg):
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


    #del r_chrom, r_break, map_positions, rates

    #data = [] #initialize list to store summary statistics 
    alpha = 1.7
    sample_size = 50
    Ne = np.random.randint(25000, 100000)
    #Ne = 1e5
    #for i in range(reps):
    ts = msprime.sim_ancestry( #constant population model with beta coalescent
        samples = sample_size,
        population_size = Ne,
        recombination_rate = rate_map,
        model=msprime.BetaCoalescent(alpha = alpha),
        #random_seed=1234,
    )

    #del rate_map

    mts = msprime.sim_mutations(ts, rate=1e-8) #simulate mutations on treekit

    

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
    S = mts.num_sites
    summary_statistics.append(S) #Fifth column is number of segregating sites
    normalized_S = mts.segregating_sites(span_normalise=True)
    summary_statistics.append(normalized_S) #Sixth column is span normalized S
    
    num_windows = 30
    pi_array = mts.diversity(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
    summary_statistics.append(np.nanmean(pi_array))
    summary_statistics.append(scipy.stats.hmean(pi_array, nan_policy = 'omit')) #Seventh column is mean pi
    summary_statistics.append(np.nanvar(pi_array)) #Eighth column is variance of pi
    summary_statistics.append(np.nanstd(pi_array))
    pi = mts.diversity()
    summary_statistics.append(pi) #Ninth column is nucleotide diversity


    afs = mts.allele_frequency_spectrum(span_normalise=False, polarised=False)

    afs_entries = [] #initialize list of afs entries. Each element corresponds to an allele's frequency as a proportion e.g. singleton, doubleton, etc.

    for x in range(1, sample_size + 2):
        num_mutations = afs[x]
        l = [x/(sample_size*2)] * int(num_mutations) #create list of allele frequency with length of number of mutations
        afs_entries.extend(l) #extend afs_entries list by elements of the new list
    afs_entries = np.array(afs_entries) 
    #len(afs_entries)
    
    summary_statistics.append(afs[1]/sum(afs))
    summary_statistics.append(afs[2]/sum(afs))
    summary_statistics.append(afs[3]/sum(afs))
    summary_statistics.append(afs[4]/sum(afs))
    summary_statistics.append(afs[5]/sum(afs))
    summary_statistics.append(afs[6]/sum(afs))
    summary_statistics.append(afs[7]/sum(afs))
    summary_statistics.append(afs[8]/sum(afs))
    summary_statistics.append(afs[9]/sum(afs))
    summary_statistics.append(afs[10]/sum(afs))
    summary_statistics.append(afs[11]/sum(afs))
    summary_statistics.append(afs[12]/sum(afs))
    summary_statistics.append(afs[13]/sum(afs))
    summary_statistics.append(afs[14]/sum(afs))
    summary_statistics.append(sum(afs[15:])/sum(afs))

    afs_quant = np.quantile(afs_entries, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(afs_quant[0]) #10th column is AFS quantile 0.1
    summary_statistics.append(afs_quant[1]) #11th column 0.3
    summary_statistics.append(afs_quant[2]) #12th column 0.5
    summary_statistics.append(afs_quant[3]) #13th column 0.7
    summary_statistics.append(afs_quant[4]) #14th column 0.9

    del afs_entries
    del afs_quant

    D_array = mts.Tajimas_D(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
    summary_statistics.append(np.nanmean(D_array)) #15th column is mean Tajima's D
    summary_statistics.append(np.nanvar(D_array)) #16th column is variance of Tajima's D
    summary_statistics.append(np.nanstd(D_array))
    del D_array

    ts_chroms = []
    for j in range(len(chrom_positions) - 1): #split genome into chromosomes
        start, end = chrom_positions[j: j + 2]
        chrom_ts = mts.keep_intervals([[start, end]], simplify=False).trim()
        ts_chroms.append(chrom_ts)
        #print(chrom_ts.sequence_length)

    

    gn = mts.genotype_matrix()
    # print("Converted to genotype matrix...")
    r = allel.rogers_huff_r(gn)

    del gn
    # print("Calculated r...")
    s = scipy.spatial.distance.squareform(r ** 2) #calculate r^2

    arr = mts.sites_position #array of site positions
    pairwise_distances = abs(arr[:, None] - arr) #broadcast subtraction to create matrix of pairwise distances between sites

    h = allel.HaplotypeArray(mts.genotype_matrix())
    #allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))
    summary_statistics.append(np.nanmean(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    summary_statistics.append(np.nanstd(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    summary_statistics.append(np.nanvar(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    hamming_distances = scipy.spatial.distance.pdist(h, metric='hamming') * h.shape[1]
    pairwise_matrix = scipy.spatial.distance.squareform(hamming_distances)
    hamming_array = pairwise_matrix[np.triu_indices_from(pairwise_matrix, k=1)]
    hamming_quant = np.quantile(hamming_array, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(hamming_quant[0]) #27th column is AFS quantile 0.1
    summary_statistics.append(hamming_quant[1]) #28th column 0.3
    summary_statistics.append(hamming_quant[2]) #29th column 0.5
    summary_statistics.append(hamming_quant[3]) #30th column 0.7
    summary_statistics.append(hamming_quant[4]) #31 column 0.9
    summary_statistics.append(np.nanmean(hamming_array)) #38 mean hamming
    summary_statistics.append(np.nanstd(hamming_array)) #39 std hamming
    summary_statistics.append(np.nanvar(hamming_array)) #40 var hamming
    #scaled_ld = np.multiply(result, s) #scale LD by distance between pairs of SNPs; matrix multiplication of distances times r^2

    del arr, ts
    #Get the lengths of homozygous runs
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_homozygous = pairwise_distances[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_homozygous = pairwise_distances[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_homozygous = pairwise_distances[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]

    

    chrom1_homozygous = chrom1_homozygous[np.triu_indices_from(chrom1_homozygous, k=1)]
    chrom2_homozygous = chrom2_homozygous[np.triu_indices_from(chrom2_homozygous, k=1)]
    chrom3_homozygous = chrom3_homozygous[np.triu_indices_from(chrom3_homozygous, k=1)]



    homozygous = np.concatenate((chrom1_homozygous, chrom2_homozygous, chrom3_homozygous))

    del chrom1_homozygous, chrom2_homozygous, chrom3_homozygous

    homozygous_quant = np.nanquantile(homozygous, [0.1,0.3,0.5,0.7,0.9])


    summary_statistics.append(homozygous_quant[0]) #17th-23rd columns lengths of homozygosity
    summary_statistics.append(homozygous_quant[1])
    summary_statistics.append(homozygous_quant[2])
    summary_statistics.append(homozygous_quant[3])
    summary_statistics.append(homozygous_quant[4])
    summary_statistics.append(np.nanmean(homozygous))
    summary_statistics.append(scipy.stats.hmean(homozygous, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(homozygous))
    summary_statistics.append(np.nanstd(homozygous))

    del homozygous, homozygous_quant, pairwise_distances
    
    #Get LD only on same chromosomes
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_ld = s[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_ld = s[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_ld = s[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]

    
    #Upper triangle of matrix to get rid of duplicated values
    chrom1_ld = chrom1_ld[np.triu_indices_from(chrom1_ld, k=1)]
    chrom2_ld = chrom2_ld[np.triu_indices_from(chrom2_ld, k=1)]
    chrom3_ld = chrom3_ld[np.triu_indices_from(chrom3_ld, k=1)]



    r2 = np.concatenate((chrom1_ld, chrom2_ld, chrom3_ld))
    r2_quant = np.nanquantile(r2, [0.1,0.3,0.5,0.7,0.9])
    
    del chrom1_ld, chrom2_ld, chrom3_ld


    summary_statistics.append(r2_quant[0]) #24th-30th columns are r^2 quantiles, mean, and variance
    summary_statistics.append(r2_quant[1])
    summary_statistics.append(r2_quant[2])
    summary_statistics.append(r2_quant[3])
    summary_statistics.append(r2_quant[4])
    summary_statistics.append(np.nanmean(r2))
    summary_statistics.append(scipy.stats.hmean(r2, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(r2))
    summary_statistics.append(np.nanstd(r2))

    del r2, r2_quant

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

    del chrom1_mut_num, chrom2_mut_num, chrom3_mut_num, chrom1and2_mut_num, total_mut_num
    ild_all = np.concatenate((chrom1_ild, chrom2_ild, chrom3_ild))
    ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9])

    del chrom1_ild, chrom2_ild, chrom3_ild

    summary_statistics.append(ild_quant[0]) #31st-37th columns ILD
    summary_statistics.append(ild_quant[1])
    summary_statistics.append(ild_quant[2])
    summary_statistics.append(ild_quant[3])
    summary_statistics.append(ild_quant[4])
    summary_statistics.append(np.nanmean(ild_all))
    summary_statistics.append(scipy.stats.hmean(ild_all, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(ild_all))
    summary_statistics.append(np.nanstd(ild_all))

    del ild_all, ild_quant
    #Append to data list to form data frame outside loop (faster than appending data frame)
    #data.append(summary_statistics)
    x = DataFrame(summary_statistics).T

    x.to_csv('summary_statistics_improved.csv', index = False, mode = 'a', header = False)
     
def alpha1_5(arg):
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


    #del r_chrom, r_break, map_positions, rates

    #data = [] #initialize list to store summary statistics 
    alpha = 1.5
    sample_size = 50
    Ne = np.random.randint(25000, 100000)
    #Ne = 1e5
    #for i in range(reps):
    ts = msprime.sim_ancestry( #constant population model with beta coalescent
        samples = sample_size,
        population_size = Ne,
        recombination_rate = rate_map,
        model=msprime.BetaCoalescent(alpha = alpha),
        #random_seed=1234,
    )

    #del rate_map

    mts = msprime.sim_mutations(ts, rate=1e-8) #simulate mutations on treekit

    

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
    S = mts.num_sites
    summary_statistics.append(S) #Fifth column is number of segregating sites
    normalized_S = mts.segregating_sites(span_normalise=True)
    summary_statistics.append(normalized_S) #Sixth column is span normalized S
    
    num_windows = 30
    pi_array = mts.diversity(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
    summary_statistics.append(np.nanmean(pi_array))
    summary_statistics.append(scipy.stats.hmean(pi_array, nan_policy = 'omit')) #Seventh column is mean pi
    summary_statistics.append(np.nanvar(pi_array)) #Eighth column is variance of pi
    summary_statistics.append(np.nanstd(pi_array))
    pi = mts.diversity()
    summary_statistics.append(pi) #Ninth column is nucleotide diversity


    afs = mts.allele_frequency_spectrum(span_normalise=False, polarised=False)

    afs_entries = [] #initialize list of afs entries. Each element corresponds to an allele's frequency as a proportion e.g. singleton, doubleton, etc.

    for x in range(1, sample_size + 2):
        num_mutations = afs[x]
        l = [x/(sample_size*2)] * int(num_mutations) #create list of allele frequency with length of number of mutations
        afs_entries.extend(l) #extend afs_entries list by elements of the new list
    afs_entries = np.array(afs_entries) 
    #len(afs_entries)
    
    summary_statistics.append(afs[1]/sum(afs))
    summary_statistics.append(afs[2]/sum(afs))
    summary_statistics.append(afs[3]/sum(afs))
    summary_statistics.append(afs[4]/sum(afs))
    summary_statistics.append(afs[5]/sum(afs))
    summary_statistics.append(afs[6]/sum(afs))
    summary_statistics.append(afs[7]/sum(afs))
    summary_statistics.append(afs[8]/sum(afs))
    summary_statistics.append(afs[9]/sum(afs))
    summary_statistics.append(afs[10]/sum(afs))
    summary_statistics.append(afs[11]/sum(afs))
    summary_statistics.append(afs[12]/sum(afs))
    summary_statistics.append(afs[13]/sum(afs))
    summary_statistics.append(afs[14]/sum(afs))
    summary_statistics.append(sum(afs[15:])/sum(afs))

    afs_quant = np.quantile(afs_entries, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(afs_quant[0]) #10th column is AFS quantile 0.1
    summary_statistics.append(afs_quant[1]) #11th column 0.3
    summary_statistics.append(afs_quant[2]) #12th column 0.5
    summary_statistics.append(afs_quant[3]) #13th column 0.7
    summary_statistics.append(afs_quant[4]) #14th column 0.9

    del afs_entries
    del afs_quant

    D_array = mts.Tajimas_D(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
    summary_statistics.append(np.nanmean(D_array)) #15th column is mean Tajima's D
    summary_statistics.append(np.nanvar(D_array)) #16th column is variance of Tajima's D
    summary_statistics.append(np.nanstd(D_array))
    del D_array

    ts_chroms = []
    for j in range(len(chrom_positions) - 1): #split genome into chromosomes
        start, end = chrom_positions[j: j + 2]
        chrom_ts = mts.keep_intervals([[start, end]], simplify=False).trim()
        ts_chroms.append(chrom_ts)
        #print(chrom_ts.sequence_length)

    

    gn = mts.genotype_matrix()
    # print("Converted to genotype matrix...")
    r = allel.rogers_huff_r(gn)

    del gn
    # print("Calculated r...")
    s = scipy.spatial.distance.squareform(r ** 2) #calculate r^2

    arr = mts.sites_position #array of site positions
    pairwise_distances = abs(arr[:, None] - arr) #broadcast subtraction to create matrix of pairwise distances between sites

    h = allel.HaplotypeArray(mts.genotype_matrix())
    #allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))
    summary_statistics.append(np.nanmean(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    summary_statistics.append(np.nanstd(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    summary_statistics.append(np.nanvar(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    hamming_distances = scipy.spatial.distance.pdist(h, metric='hamming') * h.shape[1]
    pairwise_matrix = scipy.spatial.distance.squareform(hamming_distances)
    hamming_array = pairwise_matrix[np.triu_indices_from(pairwise_matrix, k=1)]
    hamming_quant = np.quantile(hamming_array, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(hamming_quant[0]) #27th column is AFS quantile 0.1
    summary_statistics.append(hamming_quant[1]) #28th column 0.3
    summary_statistics.append(hamming_quant[2]) #29th column 0.5
    summary_statistics.append(hamming_quant[3]) #30th column 0.7
    summary_statistics.append(hamming_quant[4]) #31 column 0.9
    summary_statistics.append(np.nanmean(hamming_array)) #38 mean hamming
    summary_statistics.append(np.nanstd(hamming_array)) #39 std hamming
    summary_statistics.append(np.nanvar(hamming_array)) #40 var hamming
    #scaled_ld = np.multiply(result, s) #scale LD by distance between pairs of SNPs; matrix multiplication of distances times r^2

    del arr, ts
    #Get the lengths of homozygous runs
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_homozygous = pairwise_distances[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_homozygous = pairwise_distances[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_homozygous = pairwise_distances[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]

    

    chrom1_homozygous = chrom1_homozygous[np.triu_indices_from(chrom1_homozygous, k=1)]
    chrom2_homozygous = chrom2_homozygous[np.triu_indices_from(chrom2_homozygous, k=1)]
    chrom3_homozygous = chrom3_homozygous[np.triu_indices_from(chrom3_homozygous, k=1)]



    homozygous = np.concatenate((chrom1_homozygous, chrom2_homozygous, chrom3_homozygous))

    del chrom1_homozygous, chrom2_homozygous, chrom3_homozygous

    homozygous_quant = np.nanquantile(homozygous, [0.1,0.3,0.5,0.7,0.9])


    summary_statistics.append(homozygous_quant[0]) #17th-23rd columns lengths of homozygosity
    summary_statistics.append(homozygous_quant[1])
    summary_statistics.append(homozygous_quant[2])
    summary_statistics.append(homozygous_quant[3])
    summary_statistics.append(homozygous_quant[4])
    summary_statistics.append(np.nanmean(homozygous))
    summary_statistics.append(scipy.stats.hmean(homozygous, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(homozygous))
    summary_statistics.append(np.nanstd(homozygous))

    del homozygous, homozygous_quant, pairwise_distances
    
    #Get LD only on same chromosomes
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_ld = s[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_ld = s[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_ld = s[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]

    
    #Upper triangle of matrix to get rid of duplicated values
    chrom1_ld = chrom1_ld[np.triu_indices_from(chrom1_ld, k=1)]
    chrom2_ld = chrom2_ld[np.triu_indices_from(chrom2_ld, k=1)]
    chrom3_ld = chrom3_ld[np.triu_indices_from(chrom3_ld, k=1)]



    r2 = np.concatenate((chrom1_ld, chrom2_ld, chrom3_ld))
    r2_quant = np.nanquantile(r2, [0.1,0.3,0.5,0.7,0.9])
    
    del chrom1_ld, chrom2_ld, chrom3_ld


    summary_statistics.append(r2_quant[0]) #24th-30th columns are r^2 quantiles, mean, and variance
    summary_statistics.append(r2_quant[1])
    summary_statistics.append(r2_quant[2])
    summary_statistics.append(r2_quant[3])
    summary_statistics.append(r2_quant[4])
    summary_statistics.append(np.nanmean(r2))
    summary_statistics.append(scipy.stats.hmean(r2, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(r2))
    summary_statistics.append(np.nanstd(r2))

    del r2, r2_quant

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

    del chrom1_mut_num, chrom2_mut_num, chrom3_mut_num, chrom1and2_mut_num, total_mut_num
    ild_all = np.concatenate((chrom1_ild, chrom2_ild, chrom3_ild))
    ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9])

    del chrom1_ild, chrom2_ild, chrom3_ild

    summary_statistics.append(ild_quant[0]) #31st-37th columns ILD
    summary_statistics.append(ild_quant[1])
    summary_statistics.append(ild_quant[2])
    summary_statistics.append(ild_quant[3])
    summary_statistics.append(ild_quant[4])
    summary_statistics.append(np.nanmean(ild_all))
    summary_statistics.append(scipy.stats.hmean(ild_all, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(ild_all))
    summary_statistics.append(np.nanstd(ild_all))

    del ild_all, ild_quant
    #Append to data list to form data frame outside loop (faster than appending data frame)
    #data.append(summary_statistics)
    x = DataFrame(summary_statistics).T

    x.to_csv('summary_statistics_improved.csv', index = False, mode = 'a', header = False)
   
def alpha1_3(arg):
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


    #del r_chrom, r_break, map_positions, rates

    #data = [] #initialize list to store summary statistics 
    alpha = 1.3
    sample_size = 50
    Ne = np.random.randint(25000, 100000)
    #Ne = 1e5
    #for i in range(reps):
    ts = msprime.sim_ancestry( #constant population model with beta coalescent
        samples = sample_size,
        population_size = Ne,
        recombination_rate = rate_map,
        model=msprime.BetaCoalescent(alpha = alpha),
        #random_seed=1234,
    )

    #del rate_map

    mts = msprime.sim_mutations(ts, rate=1e-8) #simulate mutations on treekit

    

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
    S = mts.num_sites
    summary_statistics.append(S) #Fifth column is number of segregating sites
    normalized_S = mts.segregating_sites(span_normalise=True)
    summary_statistics.append(normalized_S) #Sixth column is span normalized S
    
    num_windows = 30
    pi_array = mts.diversity(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
    summary_statistics.append(np.nanmean(pi_array))
    summary_statistics.append(scipy.stats.hmean(pi_array, nan_policy = 'omit')) #Seventh column is mean pi
    summary_statistics.append(np.nanvar(pi_array)) #Eighth column is variance of pi
    summary_statistics.append(np.nanstd(pi_array))
    pi = mts.diversity()
    summary_statistics.append(pi) #Ninth column is nucleotide diversity


    afs = mts.allele_frequency_spectrum(span_normalise=False, polarised=False)

    afs_entries = [] #initialize list of afs entries. Each element corresponds to an allele's frequency as a proportion e.g. singleton, doubleton, etc.

    for x in range(1, sample_size + 2):
        num_mutations = afs[x]
        l = [x/(sample_size*2)] * int(num_mutations) #create list of allele frequency with length of number of mutations
        afs_entries.extend(l) #extend afs_entries list by elements of the new list
    afs_entries = np.array(afs_entries) 
    #len(afs_entries)
    
    summary_statistics.append(afs[1]/sum(afs))
    summary_statistics.append(afs[2]/sum(afs))
    summary_statistics.append(afs[3]/sum(afs))
    summary_statistics.append(afs[4]/sum(afs))
    summary_statistics.append(afs[5]/sum(afs))
    summary_statistics.append(afs[6]/sum(afs))
    summary_statistics.append(afs[7]/sum(afs))
    summary_statistics.append(afs[8]/sum(afs))
    summary_statistics.append(afs[9]/sum(afs))
    summary_statistics.append(afs[10]/sum(afs))
    summary_statistics.append(afs[11]/sum(afs))
    summary_statistics.append(afs[12]/sum(afs))
    summary_statistics.append(afs[13]/sum(afs))
    summary_statistics.append(afs[14]/sum(afs))
    summary_statistics.append(sum(afs[15:])/sum(afs))

    afs_quant = np.quantile(afs_entries, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(afs_quant[0]) #10th column is AFS quantile 0.1
    summary_statistics.append(afs_quant[1]) #11th column 0.3
    summary_statistics.append(afs_quant[2]) #12th column 0.5
    summary_statistics.append(afs_quant[3]) #13th column 0.7
    summary_statistics.append(afs_quant[4]) #14th column 0.9

    del afs_entries
    del afs_quant

    D_array = mts.Tajimas_D(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
    summary_statistics.append(np.nanmean(D_array)) #15th column is mean Tajima's D
    summary_statistics.append(np.nanvar(D_array)) #16th column is variance of Tajima's D
    summary_statistics.append(np.nanstd(D_array))
    del D_array

    ts_chroms = []
    for j in range(len(chrom_positions) - 1): #split genome into chromosomes
        start, end = chrom_positions[j: j + 2]
        chrom_ts = mts.keep_intervals([[start, end]], simplify=False).trim()
        ts_chroms.append(chrom_ts)
        #print(chrom_ts.sequence_length)

    

    gn = mts.genotype_matrix()
    # print("Converted to genotype matrix...")
    r = allel.rogers_huff_r(gn)

    del gn
    # print("Calculated r...")
    s = scipy.spatial.distance.squareform(r ** 2) #calculate r^2

    arr = mts.sites_position #array of site positions
    pairwise_distances = abs(arr[:, None] - arr) #broadcast subtraction to create matrix of pairwise distances between sites

    h = allel.HaplotypeArray(mts.genotype_matrix())
    #allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))
    summary_statistics.append(np.nanmean(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    summary_statistics.append(np.nanstd(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    summary_statistics.append(np.nanvar(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    hamming_distances = scipy.spatial.distance.pdist(h, metric='hamming') * h.shape[1]
    pairwise_matrix = scipy.spatial.distance.squareform(hamming_distances)
    hamming_array = pairwise_matrix[np.triu_indices_from(pairwise_matrix, k=1)]
    hamming_quant = np.quantile(hamming_array, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(hamming_quant[0]) #27th column is AFS quantile 0.1
    summary_statistics.append(hamming_quant[1]) #28th column 0.3
    summary_statistics.append(hamming_quant[2]) #29th column 0.5
    summary_statistics.append(hamming_quant[3]) #30th column 0.7
    summary_statistics.append(hamming_quant[4]) #31 column 0.9
    summary_statistics.append(np.nanmean(hamming_array)) #38 mean hamming
    summary_statistics.append(np.nanstd(hamming_array)) #39 std hamming
    summary_statistics.append(np.nanvar(hamming_array)) #40 var hamming
    #scaled_ld = np.multiply(result, s) #scale LD by distance between pairs of SNPs; matrix multiplication of distances times r^2

    del arr, ts
    #Get the lengths of homozygous runs
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_homozygous = pairwise_distances[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_homozygous = pairwise_distances[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_homozygous = pairwise_distances[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]

    

    chrom1_homozygous = chrom1_homozygous[np.triu_indices_from(chrom1_homozygous, k=1)]
    chrom2_homozygous = chrom2_homozygous[np.triu_indices_from(chrom2_homozygous, k=1)]
    chrom3_homozygous = chrom3_homozygous[np.triu_indices_from(chrom3_homozygous, k=1)]



    homozygous = np.concatenate((chrom1_homozygous, chrom2_homozygous, chrom3_homozygous))

    del chrom1_homozygous, chrom2_homozygous, chrom3_homozygous

    homozygous_quant = np.nanquantile(homozygous, [0.1,0.3,0.5,0.7,0.9])


    summary_statistics.append(homozygous_quant[0]) #17th-23rd columns lengths of homozygosity
    summary_statistics.append(homozygous_quant[1])
    summary_statistics.append(homozygous_quant[2])
    summary_statistics.append(homozygous_quant[3])
    summary_statistics.append(homozygous_quant[4])
    summary_statistics.append(np.nanmean(homozygous))
    summary_statistics.append(scipy.stats.hmean(homozygous, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(homozygous))
    summary_statistics.append(np.nanstd(homozygous))

    del homozygous, homozygous_quant, pairwise_distances
    
    #Get LD only on same chromosomes
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_ld = s[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_ld = s[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_ld = s[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]

    
    #Upper triangle of matrix to get rid of duplicated values
    chrom1_ld = chrom1_ld[np.triu_indices_from(chrom1_ld, k=1)]
    chrom2_ld = chrom2_ld[np.triu_indices_from(chrom2_ld, k=1)]
    chrom3_ld = chrom3_ld[np.triu_indices_from(chrom3_ld, k=1)]



    r2 = np.concatenate((chrom1_ld, chrom2_ld, chrom3_ld))
    r2_quant = np.nanquantile(r2, [0.1,0.3,0.5,0.7,0.9])
    
    del chrom1_ld, chrom2_ld, chrom3_ld


    summary_statistics.append(r2_quant[0]) #24th-30th columns are r^2 quantiles, mean, and variance
    summary_statistics.append(r2_quant[1])
    summary_statistics.append(r2_quant[2])
    summary_statistics.append(r2_quant[3])
    summary_statistics.append(r2_quant[4])
    summary_statistics.append(np.nanmean(r2))
    summary_statistics.append(scipy.stats.hmean(r2, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(r2))
    summary_statistics.append(np.nanstd(r2))

    del r2, r2_quant

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

    del chrom1_mut_num, chrom2_mut_num, chrom3_mut_num, chrom1and2_mut_num, total_mut_num
    ild_all = np.concatenate((chrom1_ild, chrom2_ild, chrom3_ild))
    ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9])

    del chrom1_ild, chrom2_ild, chrom3_ild

    summary_statistics.append(ild_quant[0]) #31st-37th columns ILD
    summary_statistics.append(ild_quant[1])
    summary_statistics.append(ild_quant[2])
    summary_statistics.append(ild_quant[3])
    summary_statistics.append(ild_quant[4])
    summary_statistics.append(np.nanmean(ild_all))
    summary_statistics.append(scipy.stats.hmean(ild_all, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(ild_all))
    summary_statistics.append(np.nanstd(ild_all))

    del ild_all, ild_quant
    #Append to data list to form data frame outside loop (faster than appending data frame)
    #data.append(summary_statistics)
    x = DataFrame(summary_statistics).T

    x.to_csv('summary_statistics_improved.csv', index = False, mode = 'a', header = False)
   
def alpha1_1(arg):
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


    #del r_chrom, r_break, map_positions, rates

    #data = [] #initialize list to store summary statistics 
    alpha = 1.1
    sample_size = 50
    Ne = np.random.randint(25000, 100000)
    #Ne = 1e5
    #for i in range(reps):
    ts = msprime.sim_ancestry( #constant population model with beta coalescent
        samples = sample_size,
        population_size = Ne,
        recombination_rate = rate_map,
        model=msprime.BetaCoalescent(alpha = alpha),
        #random_seed=1234,
    )

    #del rate_map

    mts = msprime.sim_mutations(ts, rate=1e-8) #simulate mutations on treekit

    

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
    S = mts.num_sites
    summary_statistics.append(S) #Fifth column is number of segregating sites
    normalized_S = mts.segregating_sites(span_normalise=True)
    summary_statistics.append(normalized_S) #Sixth column is span normalized S
    
    num_windows = 30
    pi_array = mts.diversity(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
    summary_statistics.append(np.nanmean(pi_array))
    summary_statistics.append(scipy.stats.hmean(pi_array, nan_policy = 'omit')) #Seventh column is mean pi
    summary_statistics.append(np.nanvar(pi_array)) #Eighth column is variance of pi
    summary_statistics.append(np.nanstd(pi_array))
    pi = mts.diversity()
    summary_statistics.append(pi) #Ninth column is nucleotide diversity


    afs = mts.allele_frequency_spectrum(span_normalise=False, polarised=False)

    afs_entries = [] #initialize list of afs entries. Each element corresponds to an allele's frequency as a proportion e.g. singleton, doubleton, etc.

    for x in range(1, sample_size + 2):
        num_mutations = afs[x]
        l = [x/(sample_size*2)] * int(num_mutations) #create list of allele frequency with length of number of mutations
        afs_entries.extend(l) #extend afs_entries list by elements of the new list
    afs_entries = np.array(afs_entries) 
    #len(afs_entries)
    
    summary_statistics.append(afs[1]/sum(afs))
    summary_statistics.append(afs[2]/sum(afs))
    summary_statistics.append(afs[3]/sum(afs))
    summary_statistics.append(afs[4]/sum(afs))
    summary_statistics.append(afs[5]/sum(afs))
    summary_statistics.append(afs[6]/sum(afs))
    summary_statistics.append(afs[7]/sum(afs))
    summary_statistics.append(afs[8]/sum(afs))
    summary_statistics.append(afs[9]/sum(afs))
    summary_statistics.append(afs[10]/sum(afs))
    summary_statistics.append(afs[11]/sum(afs))
    summary_statistics.append(afs[12]/sum(afs))
    summary_statistics.append(afs[13]/sum(afs))
    summary_statistics.append(afs[14]/sum(afs))
    summary_statistics.append(sum(afs[15:])/sum(afs))

    afs_quant = np.quantile(afs_entries, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(afs_quant[0]) #10th column is AFS quantile 0.1
    summary_statistics.append(afs_quant[1]) #11th column 0.3
    summary_statistics.append(afs_quant[2]) #12th column 0.5
    summary_statistics.append(afs_quant[3]) #13th column 0.7
    summary_statistics.append(afs_quant[4]) #14th column 0.9

    del afs_entries
    del afs_quant

    D_array = mts.Tajimas_D(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
    summary_statistics.append(np.nanmean(D_array)) #15th column is mean Tajima's D
    summary_statistics.append(np.nanvar(D_array)) #16th column is variance of Tajima's D
    summary_statistics.append(np.nanstd(D_array))
    del D_array

    ts_chroms = []
    for j in range(len(chrom_positions) - 1): #split genome into chromosomes
        start, end = chrom_positions[j: j + 2]
        chrom_ts = mts.keep_intervals([[start, end]], simplify=False).trim()
        ts_chroms.append(chrom_ts)
        #print(chrom_ts.sequence_length)

    

    gn = mts.genotype_matrix()
    # print("Converted to genotype matrix...")
    r = allel.rogers_huff_r(gn)

    del gn
    # print("Calculated r...")
    s = scipy.spatial.distance.squareform(r ** 2) #calculate r^2

    arr = mts.sites_position #array of site positions
    pairwise_distances = abs(arr[:, None] - arr) #broadcast subtraction to create matrix of pairwise distances between sites

    h = allel.HaplotypeArray(mts.genotype_matrix())
    #allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))
    summary_statistics.append(np.nanmean(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    summary_statistics.append(np.nanstd(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    summary_statistics.append(np.nanvar(allel.inbreeding_coefficient(h.to_genotypes(ploidy=2))))
    hamming_distances = scipy.spatial.distance.pdist(h, metric='hamming') * h.shape[1]
    pairwise_matrix = scipy.spatial.distance.squareform(hamming_distances)
    hamming_array = pairwise_matrix[np.triu_indices_from(pairwise_matrix, k=1)]
    hamming_quant = np.quantile(hamming_array, [0.1, 0.3, 0.5, 0.7, 0.9])
    summary_statistics.append(hamming_quant[0]) #27th column is AFS quantile 0.1
    summary_statistics.append(hamming_quant[1]) #28th column 0.3
    summary_statistics.append(hamming_quant[2]) #29th column 0.5
    summary_statistics.append(hamming_quant[3]) #30th column 0.7
    summary_statistics.append(hamming_quant[4]) #31 column 0.9
    summary_statistics.append(np.nanmean(hamming_array)) #38 mean hamming
    summary_statistics.append(np.nanstd(hamming_array)) #39 std hamming
    summary_statistics.append(np.nanvar(hamming_array)) #40 var hamming
    #scaled_ld = np.multiply(result, s) #scale LD by distance between pairs of SNPs; matrix multiplication of distances times r^2

    del arr, ts
    #Get the lengths of homozygous runs
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_homozygous = pairwise_distances[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_homozygous = pairwise_distances[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_homozygous = pairwise_distances[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]

    

    chrom1_homozygous = chrom1_homozygous[np.triu_indices_from(chrom1_homozygous, k=1)]
    chrom2_homozygous = chrom2_homozygous[np.triu_indices_from(chrom2_homozygous, k=1)]
    chrom3_homozygous = chrom3_homozygous[np.triu_indices_from(chrom3_homozygous, k=1)]



    homozygous = np.concatenate((chrom1_homozygous, chrom2_homozygous, chrom3_homozygous))

    del chrom1_homozygous, chrom2_homozygous, chrom3_homozygous

    homozygous_quant = np.nanquantile(homozygous, [0.1,0.3,0.5,0.7,0.9])


    summary_statistics.append(homozygous_quant[0]) #17th-23rd columns lengths of homozygosity
    summary_statistics.append(homozygous_quant[1])
    summary_statistics.append(homozygous_quant[2])
    summary_statistics.append(homozygous_quant[3])
    summary_statistics.append(homozygous_quant[4])
    summary_statistics.append(np.nanmean(homozygous))
    summary_statistics.append(scipy.stats.hmean(homozygous, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(homozygous))
    summary_statistics.append(np.nanstd(homozygous))

    del homozygous, homozygous_quant, pairwise_distances
    
    #Get LD only on same chromosomes
    chrom1_mut_num = ts_chroms[0].num_sites
    chrom1_ld = s[:chrom1_mut_num,:chrom1_mut_num]

    chrom2_mut_num = ts_chroms[1].num_sites
    chrom1and2_mut_num = chrom1_mut_num + chrom2_mut_num
    chrom2_ld = s[chrom1_mut_num:chrom1and2_mut_num, chrom1_mut_num:chrom1and2_mut_num]

    chrom3_mut_num = ts_chroms[2].num_sites
    total_mut_num = chrom1and2_mut_num + chrom3_mut_num
    chrom3_ld = s[chrom1and2_mut_num:total_mut_num,chrom1and2_mut_num:total_mut_num]

    
    #Upper triangle of matrix to get rid of duplicated values
    chrom1_ld = chrom1_ld[np.triu_indices_from(chrom1_ld, k=1)]
    chrom2_ld = chrom2_ld[np.triu_indices_from(chrom2_ld, k=1)]
    chrom3_ld = chrom3_ld[np.triu_indices_from(chrom3_ld, k=1)]



    r2 = np.concatenate((chrom1_ld, chrom2_ld, chrom3_ld))
    r2_quant = np.nanquantile(r2, [0.1,0.3,0.5,0.7,0.9])
    
    del chrom1_ld, chrom2_ld, chrom3_ld


    summary_statistics.append(r2_quant[0]) #24th-30th columns are r^2 quantiles, mean, and variance
    summary_statistics.append(r2_quant[1])
    summary_statistics.append(r2_quant[2])
    summary_statistics.append(r2_quant[3])
    summary_statistics.append(r2_quant[4])
    summary_statistics.append(np.nanmean(r2))
    summary_statistics.append(scipy.stats.hmean(r2, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(r2))
    summary_statistics.append(np.nanstd(r2))

    del r2, r2_quant

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

    del chrom1_mut_num, chrom2_mut_num, chrom3_mut_num, chrom1and2_mut_num, total_mut_num
    ild_all = np.concatenate((chrom1_ild, chrom2_ild, chrom3_ild))
    ild_quant = np.nanquantile(ild_all, [0.1,0.3,0.5,0.7,0.9])

    del chrom1_ild, chrom2_ild, chrom3_ild

    summary_statistics.append(ild_quant[0]) #31st-37th columns ILD
    summary_statistics.append(ild_quant[1])
    summary_statistics.append(ild_quant[2])
    summary_statistics.append(ild_quant[3])
    summary_statistics.append(ild_quant[4])
    summary_statistics.append(np.nanmean(ild_all))
    summary_statistics.append(scipy.stats.hmean(ild_all, nan_policy = 'omit'))
    summary_statistics.append(np.nanvar(ild_all))
    summary_statistics.append(np.nanstd(ild_all))

    del ild_all, ild_quant
    #Append to data list to form data frame outside loop (faster than appending data frame)
    #data.append(summary_statistics)
    x = DataFrame(summary_statistics).T

    x.to_csv('summary_statistics_improved.csv', index = False, mode = 'a', header = False)
    
import concurrent.futures
worker_num = 7
reps = 1
with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    list(executor.map(alpha1_9, range(reps)))

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    list(executor.map(alpha1_7, range(reps)))

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    list(executor.map(alpha1_5, range(reps)))

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    list(executor.map(alpha1_3, range(reps)))

with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
    list(executor.map(alpha1_1, range(reps)))
