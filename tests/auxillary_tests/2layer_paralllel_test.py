'''
Test if Ray works on MPI
Task: On each MPI process, try multi-threading with Ray where each thread sleeps for 10s
'''

import numpy as np
from mpi4py import MPI
import ray

@ray.remote
def square(number):
    return number ** 2

# Perform square calculations in parallel
results = ray.get([square.remote(i) for i in range(1, 6)])
print(results)  # Output: [1, 4, 9, 16, 25]

