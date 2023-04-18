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
                if len(X_is[i]) < (1 - eps) * len(X):
                    i_star = i
                    break

            S = S.union(a[: i_star + 1])
            X = X_is[i_star]

    return S

def modified_ThresholdGreedy(f, S, G, tau, k, eps):
    '''
        Function:
            Run Adaptive Sequencing algorithm on f
        
        Input:
            f: submodular function
            S: input set
            G: partial greedy solution
            tau: threshold
            k: integer, cardinality constraint
    '''

    X = S.copy()
    G_prime = G.copy()

    while len(X) != 0 and len(G_prime) < k:

        a = random_sequence(k, G_prime, X)
        X_is = []

        # Calculate X_i
        for i in range(len(a)):

            G_i = G_prime.union(set(a[:i+1]))
            X_i = find_X_i(G_i, f, X, tau)
            X_is.append(X_i)

        # Find i*
        for i in range(len(X_is)):
            if len(X_is[i]) <= (1 - eps) * len(X):
                i_star = i
                break

        G_prime = G_prime.union(a[: i_star + 1])
        X = X_is[i_star]

    return G_prime

def two_round_MapReduce(f, k, V, OPT, m):
    '''
        Function:
            Run two round MapReduce on f

        Input:
            f: submodular function
            k: integer, cardinality constraint
            V: set, the universe
            OPT: estimated optimal value
            m: integer, number of machines

    '''

    p = 4 * np.sqrt(k/len(V))
    S, V_is = PartitionAndSample(V, m, p)

    # Should be run on each machine
    # Need modification when run on distributed system
    tau = 0.5 * OPT/k
    G_0 = ThresholdGreedy(f, S, set(), tau, k)

    if len(G_0) < k:

        R = set()
        
        for V_i in V_is:
            print('checkV_i')
            R_i = ThresholdFilter(f, V_i, G_0, tau)
            R.union(R_i)
        
        return ThresholdGreedy(f, R, G_0, tau, k)

    else:

        return G_0

def modified_MapReduce(f, k, V, OPT, m, eps):
    '''
        Function:
            Run two round MapReduce on f

        Input:
            f: submodular function
            k: integer, cardinality constraint
            V: set, the universe
            OPT: estimated optimal value
            m: integer, number of machines

    '''

    p = 4 * np.sqrt(k/len(V))
    S, V_is = PartitionAndSample(V, m, p)

    # Should be run on each machine
    # Need modification when run on distributed system
    tau = 0.5 * OPT/k
    G_0 = modified_ThresholdGreedy(f, S, set(), tau, k, eps)

    if len(G_0) < k:

        R = set()
        
        for V_i in V_is:
            print('checkV_i')
            R_i = ThresholdFilter(f, V_i, G_0, tau)
            R.union(R_i)
        
        return modified_ThresholdGreedy(f, R, G_0, tau, k, eps)

    else:

        return G_0


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
            G_prime.add(e)

    return G_prime

# Need Parallel
def ThresholdFilter(f, S, G, tau):
    S_prime = S.copy()

    for e in S:
        if f(G,e) < tau:
            S_prime = S_prime - {e}

    return S_prime

# Need Parallel
def PartitionAndSample(V, m, p):
    '''
        Function:
            S = Sample every element in V with prob p. Partition V into m subsets.
    '''

    S = set()
    probs_S = rng.uniform(size = len(V))
    probs_V_i = rng.integers( m, size = len(V))

    V_is = [set() for _ in range(m)]

    for i, element in enumerate(V):
        if probs_S[i] <= p:
            S.add(element)
        
        V_is[probs_V_i[i]].add(element)
        
    return S, V_is

