import numpy as np
from mpi4py import MPI

rng = None

def two_round_MapReduce(f, k, V, OPT, eps):
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

    G_0 = ModThresholdGreedy(f, S, set(), tau, k, eps, comm)

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
    G = ModThresholdGreedy(f, R, G_0, tau, k, eps, comm)

    if rank == 0:

        end_time = MPI.Wtime()
        duration = end_time - start_time

        round2_time = end_time - time2
        print('Round2time:', round2_time)

        return G, duration

    return None, None


def ModThresholdGreedy(f, X, G_prime, tau, k, eps, comm):
    '''
        Function:
            Implements Algorithm 1 from Liu and Vondrak
        
        Input:  
            f: submodular function
            X: set
            G_prime: set, partial greedy solution
            tau: float, threshold
            k: int, cardinality constraint
    '''

    rank = comm.Get_rank()
    size = comm.Get_size()

    while len(X)!= 0 and len(G_prime) <k:



        a = None
        if comm.rank == 0:
            a = random_sequence(k, G_prime, X)
            inner_iter_start_time = MPI.Wtime()

        a = comm.bcast(a, root = 0)

        # Calculate X_i & Exponential Binary Search

        left = 0
        right = 0
        exp_index = 0

        num_rounds = 0

        # Find initial range
        while right < len(a):

            i = right

            # Distribute data
            chunk_size = len(X) // size
            remainder = len(X) % size
            list_X = list(X)

            if rank < remainder:
                local_X = list_X[ rank*(chunk_size + 1) : (rank + 1)*(chunk_size+1) ]

            else:
                local_X = list_X[ rank * chunk_size + remainder: (rank+1) * chunk_size + remainder ]

                
            G_prime_i = G_prime.union(set( a[:i+1]))
            local_candidates, num_local_candidates = find_local_candidates(G_prime_i, f, local_X, tau)

            num_all_candidates = comm.gather(num_local_candidates, root = 0)
                
            X_i_size = None
            if rank ==0:
                X_i_size = sum(num_all_candidates)

            X_i_size = comm.bcast(X_i_size)
                
            num_rounds = num_rounds + 1

            if X_i_size >= (1 - eps) * len(X):
                left = right
                right = 2 ** exp_index
                exp_index += 1
            else:
                break

        # if right = 0,1,2, i_star equals right
        if right <= 2:
            left = right

        # if right is too big, scale down
        if right > len(a):
            right = len(a)-1
            
        # Binary Search
        while left < right:

            i = (left + right) // 2

            # Distribute data
            chunk_size = len(X) // size
            remainder = len(X) % size
            list_X = list(X)

            if rank < remainder:
                local_X = list_X[ rank*(chunk_size + 1) : (rank + 1)*(chunk_size+1) ]

            else:
                local_X = list_X[ rank * chunk_size + remainder: (rank+1) * chunk_size + remainder ]

                
            G_prime_i = G_prime.union(set( a[:i+1]))
            local_candidates, num_local_candidates = find_local_candidates(G_prime_i, f, local_X, tau)

            num_all_candidates = comm.gather(num_local_candidates, root = 0)
                
            X_i_size = None
            if rank ==0:
                X_i_size = sum(num_all_candidates)

            X_i_size = comm.bcast(X_i_size)

            num_rounds = num_rounds + 1
                
            if X_i_size < (1 - eps) * len(X):
                right = i
            else:
                left = i + 1
                i = left 

        G_prime.update(a[: i + 1])
            

        X_i = comm.gather(local_candidates, root = 0)

        if rank == 0:
           
            X = set()
            for portion in X_i:
                X.update(portion)

        X = comm.bcast(X, root = 0)

        if rank == 0:
            inner_iter_end_time = MPI.Wtime()
            inner_iter_time = inner_iter_end_time - inner_iter_start_time
            print( f'i:{i}, num_rounds:{num_rounds}, size_of_X:{len(X)}, duration:{inner_iter_time}')

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


def find_local_candidates(S_i, f, X, t):
    '''
        Function: X_i = elements from X which has marginal contribution to S over t
    '''

    threshold = f(S_i) + t

    X_i = set()
    for a in X:
        if f(S_i.union({a})) >= threshold:
            X_i.add(a)

    return X_i, len(X_i)