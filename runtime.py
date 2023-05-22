import optimization_algo as oa
import coverage_function as cf

import time
import numpy as np
import matplotlib.pyplot as plt

# Size of Set A
N_size = 200

# Size of Set B
max_cover = 5000

# Specify seed here
seed = 1
oa.rng = np.random.default_rng(seed)
cf.rng = np.random.default_rng(seed)

# List of Times for Each Algorithm
GR_time = []
AS_time = []
TRM_time = []
MM_time = []

# List of Cardinality constraints for testing
card_constraints = np.linspace(10, 20, num = 10, dtype = int)

for k in card_constraints:

    print(k)

    coverage, opt_elements = cf.generate_coverage(N_size, max_cover, k)
    f = lambda A, a = None : cf.cov_func(coverage, A, a)
    OPT = f(opt_elements)

    # Time for Gredy
    start = time.time()
    GR_solution = oa.greedy(f = f, k = k, N = set(range(N_size)))
    end = time.time()
    GR_time.append(end - start)

    # Time for Adaptive Sequencing
    start = time.time()
    AS_solution = oa.adaptive_sequencing(f = f, k = k, N = set(range(N_size)), 
    OPT = OPT, eps = 0.05)
    end = time.time()
    AS_time.append(end - start)

    # Time for Two-round Map-Reduce
    start = time.time()
    TRM_solution = oa.two_round_MapReduce(f = f, k = k, V = set(range(N_size)), 
    OPT = OPT, m = 5)
    end = time.time()
    TRM_time.append(end - start)

    # Time for Modified Map Reduce
    start = time.time()
    MM_solution = oa.modified_MapReduce(f = f, k = k, V = set(range(N_size)), 
    OPT = OPT, m = 5, eps = 0.05)
    end = time.time()
    MM_time.append(end - start)

size = "16" 

plt.figure(figsize=(20,10))

plt.plot(card_constraints, GR_time, label = "Greedy")
plt.plot(card_constraints, AS_time, label = "Adaptive Sequencing")
plt.plot(card_constraints, TRM_time, label = "2-round MapReduce")
plt.plot(card_constraints, MM_time, label = "Modified MapReduce")

plt.xticks(fontsize = size)
plt.yticks(fontsize = size)

plt.xlabel("Cardinality Constraint $(k)$", fontsize = size)
plt.ylabel("Runtime $(s)$", fontsize = size)

plt.title("Comparison of Submodular Maximization Algorithms (Sequential) on Coverage Function with Bipartite Graphs ($|A| = 200, |B| = 5000$)", fontsize = size)
plt.legend(fontsize = size)
plt.show()