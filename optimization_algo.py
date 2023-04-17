## The optimization algorithms: Greedy, Adaptive Sequencing, etc

import numpy as np
import icecream as ic

rng = None

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
        X = N
        t = (1-eps)*(OPT-f(S))/k

        while len(X)!= 0 and len(S) <k:

            a = random_sequence(k, S, X)
            X_is = []

            # Calculate X_i
            for i in range(len(a)):

                S_i = S.union(set( a[:i+1]))
                X_i = find_X_i(S_i, f, X, t)
                X_is.append(X_i)

            # Find i*
            for i in range(len(X_is)):
                if len(X_is[i]) < (1 - eps) * len(N):
                    i_star = i
                    break

            S = S.union(a[: i_star + 1])
            X = X_is[i_star]

    return S

def Vondrak():
    pass

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

    surviving_X = list(X - S)
    num_sample = k - len(S)

    if num_sample > len(surviving_X):
        return surviving_X
    
    samples = rng.choice(surviving_X, num_sample, replace=False)

    return samples

# Need Parallel
def ThresholdGreedy(f, S, G, tau, k):

    G_prime = G.copy()

    for e in S:
        if f(G,e) >= tau and len(G_prime) < k:
            G_prime = G_prime + {e}

# Need Parallel
def ThresholdFilter(f, S, G, tau):
    S_prime = S.copy()

    for e in S:
        if f(G,e) < tau:
            S_prime = S_prime - {e}

    return S_prime

def PartitionAndSample(V):

    S = set()

    pass

