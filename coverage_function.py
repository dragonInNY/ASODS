## Generate data -- data structures where coverage functions can be run on

import numpy as np
rng = np.random.default_rng(24601)

def generate_data(A_size, B_size, card_constraint):
    '''
    Function: 
        Coverage functions map from elements in A to N, i.e. # of elements in B. 
        Construct a mapping from A to B so that coverage funcs can be evaluated.

    Input: 
        A_size, B_size
        card_constraint: cardinality constraint

    Output: 
        coverage: a dictionary. Key: elements in A; value: corresponding elements in B.
        opt_elements: set of elements in A that cover the most elements in B
    '''
    
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
    '''
        Function:
            Split the input set into some number of subsets. 
            Note: num_subsets * size_subset <= size of input_set

    '''

    output_sets = []
    for i in range(num_subsets):
        output_sets.append(set(rng.choice(list(input_set), size_subset)))
        input_set -= output_sets[-1]
    return output_sets

def coverage_func(A, coverage):
    '''
        Function:
            Evalute a subset of A based on a coverage relationship
    '''
    covered_set = set()
    for a in A:
        covered_set = covered_set.union(coverage[a])
    return len(covered_set)

coverage, opt_ele = generate_data(5, 10, 3)
print(coverage)
print(opt_ele)
print(coverage_func(opt_ele, coverage))
