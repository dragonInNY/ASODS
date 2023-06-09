import optimization_algo as oa
import coverage_function as cf

import numpy as np
import matplotlib.pyplot as plt

AS_performance = []
TRM_performance = []

for N_size in range(20, 60, 10):

    max_cover = N_size * 10
    k = int(N_size /3)

    for seed in range(1000):
        oa.rng = np.random.default_rng(seed)
        cf.rng = np.random.default_rng(seed)

        coverage, opt_elements = cf.generate_coverage(N_size, max_cover, k)
        f = lambda A, a = None : cf.cov_func(coverage, A, a)
        OPT = f(opt_elements)

        # greedy_solution = oa.greedy(f = f, k = k, N = set(range(N_size)))
        # print(f(greedy_solution)/OPT)

        TRM_solution = oa.two_round_MapReduce(f = f, k = k, V = set(range(N_size)), OPT = OPT, m = 3)
        AS_solution = oa.adaptive_sequencing(f = f, k = k, N = set(range(N_size)), OPT = OPT, eps = 0.05)

        AS_performance.append(f(AS_solution)/OPT)
        TRM_performance.append(f(TRM_solution)/OPT)


    # Visualization

    plt.hist(AS_performance, density=True, bins=15, alpha=0.5, label='AS')
    plt.hist(TRM_performance, density=True, bins=15, alpha=0.5, label='TRM')
    plt.xlabel('Approx')
    plt.ylabel('Frequency')
    plt.title('AS = {}, TRM = {}'.format(np.mean(AS_performance), np.mean(TRM_performance)))
    plt.legend()

    save_path = '/Users/dragonyuan/Desktop/TRM_vs_AS/'
    plt.savefig( save_path + 'N={},C={},k={}'.format(N_size, max_cover, k) + '.png')
    plt.clf()
