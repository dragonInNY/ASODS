'''
Test Modified Two-round MapReduce
'''

import numpy as np
import pickle
import time
import sys
from mpi4py import MPI
import cProfile

sys.path.append('/rigel/home/hy2726/codes/opt_algos') 
import mod_TRMR
mod_TRMR.rng = np.random.default_rng(10)

sys.path.append('/rigel/home/hy2726/codes/submodular_func')
import coverage_function as cf

with open('/rigel/home/hy2726/codes/tests/stanford.pkl', "rb") as file:
    coverage = pickle.load(file)
N = set(coverage.keys())
f = lambda A, a = None : cf.cov_func(coverage, A, a)

k = 100
OPT = 164036
eps = 0.25

S, time = mod_TRMR.two_round_MapReduce(f, k, N, OPT, eps)

if MPI.COMM_WORLD.Get_rank() == 0:
    print('Duration:', time)
    print('Approximation:', f(S)/OPT)
    print(len(S))