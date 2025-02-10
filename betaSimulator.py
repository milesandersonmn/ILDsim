#! /home/milesanderson/miniconda3/envs/msprime/bin/python

import msprime
import numpy as np
import matplotlib.pyplot as plt
import math
import tskit

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

alpha = np.random.uniform(low=1.05, high=2) #Draw alpha parameter from uniform distribution
Ne = np.random.uniform(low=1000, high=1000000)
ts = msprime.sim_ancestry(
    samples=38,
    population_size=10000,
    recombination_rate=rate_map,
    model=msprime.BetaCoalescent(alpha=alpha),
    random_seed=1234,
)
ts

mts = msprime.sim_mutations(ts, rate=1e-8, random_seed=5678)

np.set_printoptions(legacy="1.21")
summary_statistics = [] #Initialize list of summary statistics
summary_statistics.append(1) #First column corresponds to model index
summary_statistics.append(10000) #Second column is Ne
summary_statistics.append(alpha) #Third column is alpha parameter
summary_statistics.append(1) #Fourth column is rho/theta
S = mts.get_num_mutations()
summary_statistics.append(S) #Fifth column is number of segregating sites
normalized_S = mts.segregating_sites(span_normalise=True)
summary_statistics.append(normalized_S) #Sixth column is span normalized S
pi = mts.diversity()
summary_statistics.append(pi) #Seventh column is nucleotide diversity
summary_statistics

afs = mts.allele_frequency_spectrum(span_normalise=False, polarised=False)

afs_entries = []

for x in range(1, 40):
   num_mutations = afs[x]
   l = [x/76] * int(num_mutations)
   afs_entries.extend(l)
afs_entries = np.array(afs_entries)

afs_quant = np.quantile(afs_entries, [0.1, 0.3, 0.5, 0.7, 0.9])
summary_statistics.append(afs_quant[0]) #8th column is AFS quantile 0.1
summary_statistics.append(afs_quant[1]) #9th column 0.3
summary_statistics.append(afs_quant[2]) #10th column 0.5
summary_statistics.append(afs_quant[3]) #11th column 0.7
summary_statistics.append(afs_quant[4]) #12th column 0.9
summary_statistics

num_windows = 30
D_array = mts.Tajimas_D(windows=np.linspace(0, ts.sequence_length, num_windows + 1))
summary_statistics.append(np.nanmean(D_array))
summary_statistics.append(np.nanvar(D_array))
summary_statistics

ts_chroms = []
for j in range(len(chrom_positions) - 1):
    start, end = chrom_positions[j: j + 2]
    chrom_ts = mts.keep_intervals([[start, end]], simplify=False).trim()
    ts_chroms.append(chrom_ts)
    print(chrom_ts.sequence_length)

ld_calc = tskit.LdCalculator(ts_chroms[0])
r2_chrom1 = ld_calc.r2_matrix()
r2_chrom1 = np.matrix.flatten(r2_chrom1)
ld_calc = tskit.LdCalculator(ts_chroms[1])
r2_chrom2 = ld_calc.r2_matrix()
r2_chrom2 = np.matrix.flatten(r2_chrom2)
ld_calc = tskit.LdCalculator(ts_chroms[2])
r2_chrom3 = ld_calc.r2_matrix()
r2_chrom3 = np.matrix.flatten(r2_chrom3)
r2 = np.concatenate((r2_chrom1,r2_chrom2,r2_chrom3))
r2_quant = np.quantile(r2, [0.1,0.3,0.5,0.7,0.9])
r2_quant

summary_statistics.append(r2_quant[0])
summary_statistics.append(r2_quant[1])
summary_statistics.append(r2_quant[2])
summary_statistics.append(r2_quant[3])
summary_statistics.append(r2_quant[4])
summary_statistics.append(np.mean(r2))
summary_statistics.append(np.var(r2))
summary_statistics

for x in range(3):
    with open("output"+str(x+1)+".vcf", "w") as vcf_file:
        ts_chroms[x].write_vcf(vcf_file, contig_id=str(x+1))

import subprocess
import os
os.getcwd()

subprocess.run(["/usr/bin/bcftools", "concat", "output1.vcf", "output2.vcf", "output3.vcf", "-o", "concat.vcf"])

subprocess.run(["/home/milesanderson/softwares/GWLD/GWLD-C++/bin/GWLD", "-v", "concat.vcf", "-c", "-o", "glwd", "-m", "RMI"])

