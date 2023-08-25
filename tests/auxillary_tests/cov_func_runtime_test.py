'''
Test if f_S(a) runs slower than f(a)
'''

import pickle
import sys
import time
import random

sys.path.append('/rigel/home/hy2726/codes/submodular_func')
import coverage_function as cf

with open('/rigel/home/hy2726/codes/tests/twitter.pkl', "rb") as file:
    coverage = pickle.load(file)

N = set(coverage.keys())
f = lambda A, a = None : cf.cov_func(coverage, A, a)

start_time = time.time()
for a in N:
    b = f(a)

end_time = time.time()

print(f'No_ground_set:{end_time - start_time}')

S = random.sample(N, 100)
S = set(S)

start_time = time.time()
for a in N:
    b = f(S, a)
end_time = time.time()

print(f'With_ground_set:{end_time - start_time}')
