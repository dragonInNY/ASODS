'''
Test AS
'''

import numpy as np
import time
import sys
import pickle
from mpi4py import MPI

sys.path.append('/rigel/home/hy2726/codes/opt_algos') 
import AS
AS.rng = np.random.default_rng(10)

sys.path.append('/rigel/home/hy2726/codes/submodular_func')
import coverage_function as cf

# # Randomly generated coverage
# N_size = 1000
# max_cover = 3000
# k = 100
# N = set(range(N_size))
# cf.rng = np.random.default_rng(10)

# coverage, opt_elements = cf.generate_coverage(N_size, max_cover, k)
# f = lambda A, a = None : cf.cov_func(coverage, A, a)
# OPT = f(opt_elements)
# eps = 0.05

# Real-world dataset coverage
with open('/rigel/home/hy2726/codes/tests/twitter.pkl', "rb") as file:
    coverage = pickle.load(file)

N = set(coverage.keys())
f = lambda A, a = None : cf.cov_func(coverage, A, a)

k = 500
OPT = 71661
eps = 0.25

S, time = AS.adaptive_sequencing(f, k, N, OPT, eps)

if MPI.COMM_WORLD.Get_rank() == 0:
    print('Duration:', time)
    print('Approximation:', f(S)/OPT)
    # print(S)
    # print(opt_elements)



