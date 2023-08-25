'''
Test greedy
'''

import numpy as np
import time
import sys
from mpi4py import MPI
import pickle

sys.path.append('/rigel/home/hy2726/codes/opt_algos') 
import greedy_lazy as greedy

sys.path.append('/rigel/home/hy2726/codes/submodular_func')
import coverage_function as cf


# # Randomly generated coverage

# N_size = 1000
# max_cover = 3000
# k = 100
# N = set(range(N_size))

# cf.rng = np.random.default_rng(24601)
# coverage, opt_elements = cf.generate_coverage(N_size, max_cover, k)


# Real-world dataset coverage

with open('/rigel/home/hy2726/codes/tests/stanford.pkl', "rb") as file:
    coverage = pickle.load(file)

N = set(coverage.keys())
f = lambda A, a = None : cf.cov_func(coverage, A, a)

k = 100

S, time = greedy.greedy(f, k, N)

if MPI.COMM_WORLD.Get_rank() == 0:
    print('Time:', time)
    print('Greedy Performance:', f(S))

    # print(S)
    # print(opt_elements)



