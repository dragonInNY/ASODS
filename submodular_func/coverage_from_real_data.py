## create_coverage from real-world dataset

import numpy as np

def create_coverage(edges: np.ndarray):

    '''
    Function: 
        Create coverage based on real-world dataset
    '''

    coverage = {}

    for u, v in edges:

        coverage[u] = coverage[u].union({v}) if coverage[u] != None else {v}
        coverage[v] = coverage[v].union({u}) if coverage[v] != None else {u}
    
    return coverage

phys_data = np.loadtxt('/rigel/home/hy2726/codes/submodular_func/ca-AstroPh.txt').astype(int) 
coverage = create_coverage(phys_data)
