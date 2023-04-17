## Generate data -- data structures where coverage functions can be run on

import numpy as np
rng = None

def generate_coverage(A_size, B_size, card_constraint):
    '''
    Function: 
        Construct a coverage with known optimal solution

    Input: 
        A_size, B_size: sizes of the two sets
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

    opt_elements = rng.choice(range(A_size), card_constraint, replace = False)
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

def cov_func(coverage, A, a = None):
    '''
        Function:
            One input: evalute a set based on a coverage. 
            Two inputs: evalute marginal contribution of a to A, where a can be set/element
    '''

    if a is not None:
        return marg_cov_func(coverage, A, a)
    
    covered_set = set()
    for a in A:
        covered_set = covered_set.union(coverage[a])
    return len(covered_set)

def marg_cov_func( coverage, A, a):
    '''
        Function:
            Evalute marginal contribution f_A(a)
    '''

    if isinstance(a, set):
        return cov_func(coverage, A.union(a)) - cov_func(coverage, A) 
    
    return cov_func(coverage, A.union({a})) - cov_func(coverage, A) 

# Testing:
# coverage, opt_ele = generate_coverage(5, 10, 3)
# print(coverage)
# print(opt_ele)
# print(cov_func(coverage, opt_ele))

# A = {4,3}
# a = {2}
# print(marg_cov_func(coverage, A, a))
# print( cov_func(coverage, A, a))
