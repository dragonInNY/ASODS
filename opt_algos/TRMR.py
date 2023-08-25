import numpy as np
import random
from mpi4py import MPI

rng = None

def two_round_MapReduce(f, k, V, OPT):
    '''
        Function:
            Implements Algorithm 4 from Liu and Vondrak
            On distributed Systems

        Input:
            f: submodular function
            k: integer, cardinality constraint
            V: set, the universe
            OPT: estimated optimal value
    '''

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        start_time = MPI.Wtime()

    p = 4 * np.sqrt(k / len(V))

    # Round 1
    S = None
    V_i = None

    if rank == 0:
        S, V_i = PartitionAndSample(V, size, p)
    
    V_i = comm.scatter(V_i, root = 0)
    S = comm.bcast(S, root = 0)
    
    tau = 0.5 * OPT/k
    # G_0 = ThresholdGreedy(f, S, set(), tau, k)
    G_0 = parallel_ThresholdGreedy(f, S, set(), tau, k, comm)

    if rank == 0:
        time1 = MPI.Wtime()

        TG_time = time1 - start_time

        print('TGtime:', TG_time)
        print('G_0 size:', len(G_0))

    R_i = ThresholdFilter(f, V_i, G_0, tau) if len(G_0) < k else set()

    R_is = comm.gather(R_i, root = 0)

    # Round 2
    R = None
    if rank == 0:

        time2 = MPI.Wtime()
        round1_time = time2 - time1
        print('TFtime:', round1_time)

        R = set()
        for r_i in R_is:
            R.update(r_i)

        print('R size:', len(R))

    R = comm.bcast(R, root = 0)
    # G = ThresholdGreedy(f, R, G_0, tau, k)
    G = parallel_ThresholdGreedy(f, R, G_0, tau, k, comm)

    if rank == 0:

        end_time = MPI.Wtime()
        duration = end_time - start_time

        round2_time = end_time - time2
        print('Round2time:', round2_time)

        return G, duration

    return None, None


def ThresholdGreedy(f, S, G, tau, k):
    '''
        Function:
            Implements Algorithm 1 from Liu and Vondrak
            Iteratively add elements from S to G with marginal contribution >= tau
        
        Input:  
            f: submodular function
            S: set
            G: set, partial greedy solution
            tau: float, threshold
            k: int, cardinality constraint
    '''

    G_prime = G.copy()

    for e in S:

        if f(G_prime, e) >= tau and len(G_prime) < k:
            G_prime.add(e)

    return G_prime

def ThresholdFilter(f, S, G, tau):
    '''
        Function:
            Implements Algorithm 2 from Liu and Vondrak
        
        Input:  
            f: submodular function
            S: set
            G: set, partial greedy solution
            tau: float, threshold
    '''

    S_prime = S.copy()

    threshold = f(G) + tau

    for e in S:
        if f(G.union({e})) < threshold:
            S_prime.remove(e)

    return S_prime

def PartitionAndSample(V, m, p):
    '''
        Function:
            Implements Algorithm 3 from Liu and Vondrak
        
        Input:  
            V: set, the universe
            m: integer, number of machines
            p: sampling probability

        Output:
            V_is: list of sets
            S: set
    '''

    S = set()
    probs_S = rng.uniform(size = len(V))
    probs_V_i = rng.integers(m, size = len(V))

    V_is = [set() for _ in range(m)]

    for i, element in enumerate(V):
        if probs_S[i] <= p:
            S.add(element)
        else:
            V_is[probs_V_i[i]].add(element)
        
    return S, V_is

def parallel_ThresholdGreedy(f, S, G, tau, k, comm):
    '''
        Function:
            Iteratively add elements from S to G with marginal contribution >= tau
        
        Input:  
            f: submodular function
            S: set
            G: set, partial greedy solution
            tau: float, threshold
            k: int, cardinality constraint
    '''

    rank = comm.Get_rank()
    size = comm.Get_size()

    G_prime = G.copy()
    X = S.copy()

    while len(G_prime) < k and len(X) > 0:

        # Partition X & distribute data
        chunk_size = len(X) // size
        remainder = len(X) % size
        list_X = list(X)

        if rank < remainder:
            local_X = list_X[ rank*(chunk_size + 1) : (rank + 1)*(chunk_size+1) ]

        else:
            local_X = list_X[ rank * chunk_size + remainder: (rank+1) * chunk_size + remainder ]

        threshold = f(G_prime) + tau
        local_candidates = set()

        # Select candidates
        for e in local_X:
            if f(G_prime.union({e})) >= threshold:
                local_candidates.add(e)

        all_candidates = comm.gather(local_candidates, root = 0)

        selected_candidate = None
        if rank == 0:
            X = set()
            for candidate in all_candidates:
                X.update(candidate)
            
            if len(X) != 0:
                
                selected_candidate = random.choice(list(X))
                G_prime.add(selected_candidate)
                X.remove(selected_candidate)
            
        X = comm.bcast(X)
        G_prime = comm.bcast(G_prime)
        
    return G_prime