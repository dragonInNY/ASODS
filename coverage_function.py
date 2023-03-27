import numpy as np
rng = np.random.default_rng(24601)


def generate_data(A_size, B_size, card_constraint):
    
    A = set(range(A_size))
    B = set(range(B_size))
    max_size = B_size//card_constraint

    coverage = {}

    for a in A:
        subset_size = rng.integers(low = 1, high = max_size)
        subset = rng.choice(range(B_size), subset_size)
        coverage[a] = set(subset)

    opt_elements = rng.choice(range(A_size), card_constraint)
    B_split = split_set(B, card_constraint, max_size)

    for i, opt_element in enumerate(opt_elements):
        coverage[opt_element] = B_split[i]

    return coverage, opt_elements

def split_set(input_set, num_subsets, size_subset):

    output_sets = []
    for i in range(num_subsets):
        output_sets.append(set(rng.choice(list(input_set), size_subset)))
        input_set -= output_sets[-1]
    return output_sets

def coverage_func(A, coverage):
    covered_set = set()
    for a in A:
        covered_set.union(coverage[a])
    return len(covered_set)

coverage, opt_ele = generate_data(5, 10, 3)
print(coverage)
print(opt_ele)
