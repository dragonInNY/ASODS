import optimization_algo as oa
import coverage_function as cf

import numpy as np
import matplotlib.pyplot as plt

performance = []

for seed in range(1000):
    oa.rng = np.random.default_rng(seed)
    cf.rng = np.random.default_rng(seed)

    N_size = 50
    max_cover = 500
    k = 20
    coverage, opt_elements = cf.generate_coverage(N_size, max_cover, k)
    f = lambda A, a = None : cf.cov_func(coverage, A, a)
    OPT = f(opt_elements)

    # greedy_solution = oa.greedy(f = f, k = k, N = set(range(N_size)))
    # print(f(greedy_solution)/OPT)

    AS_solution = oa.adaptive_sequencing(f = f, k = k, N = set(range(N_size)), OPT = OPT, eps = 0.05)

    performance.append(f(AS_solution)/OPT)

plt.hist(performance, density=True, bins=15)
print(np.mean(performance))
plt.show()