## The optimization algorithms: Greedy, Adaptive Sequencing, etc

import coverage_function as cf
import numpy as np
rng = np.random.default_rng(24601)

coverage, opt_elements = cf.generate_coverage(5, 10, 3)
f = lambda A, a = None : cf.cov_func(coverage, A, a)

def greedy(f, k, N):
    '''
        Function:
            Run greedy algorithm on f
        
        Input:
            f: submodular function
            k: integer, cardinality constraint
            N: set, the universe
    '''
    S = set()
    while len(S) < k:
        X_s = None
        f_s = 0
        for x in N:
            if f(S, x) > f_s:
                X_s = x
                f_s = f(S, x)
        S.add(X_s)
        N.remove(X_s)

    return S

def adaptive_sequencing(f, k, N, OPT, eps):
    '''
        Function:
            Run Adaptive Sequencing algorithm on f
        
        Input:
            f: submodular function
            k: integer, cardinality constraint
            N: set, the universe
            OPT: estimated optimal value
    '''
    S = set()
    iter = 0

    while len(S)<k and iter < 1/eps:
        X = N, 
        t = (1-eps)*(OPT-f(S))/k

        while len(X)!= 0 and len(S) <k:
            a = random_sequence(k, S, X)
            X_is = np.zeros(len(a))

            # Calculate X_i
            for i in range(len(a)):

                S_i = S.union(set( a[:i+1]))
                X_i = find_X_i(S_i, f, X, t)
                X_is[i] = X_i

            # Find i*
            for i in range(len(X_is)):
                if len(X_is[i]) < (1 - eps) * len(N):
                    i_star = i
                    break

            S = S.union(a[: i_star + 1])
            X = X_is[i_star]

    return S

# Auxiliary Functions

# Need Parallel
def find_X_i(S_i, f, X, t):

    X_i = set()
    for a in X:
        if f(S_i, a) >= t:
            X_i.add(a)

    return X_i

# Need Parallel
def random_sequence(k, S, X):
    '''
        Function:
            Generates a random sequence by picking k-|S| elements from X
        
        Input:
            k: integer
            S, X: set
        
        Output:
            samples: np.array
    
    '''
    num_sample = k - len(S)
    samples = rng.choice(list(X), num_sample, replace=False)

    return samples

#Testing Greedy
print(coverage)
print(greedy(f, 3, set(range(5))))
