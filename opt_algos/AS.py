import numpy as np
from mpi4py import MPI

rng = None

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

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        start_time = MPI.Wtime()

    while len(S)<k and iter < 1/eps:
        X = N.copy()
        t = (1-eps)*(OPT-f(S))/k

        while len(X)!= 0 and len(S) <k:

            a = None
            if rank == 0:
                a = random_sequence(k, S, X)

            a = comm.bcast(a, root = 0)

            # Calculate X_i & Exponential Binary Search

            left = 0
            right = 0
            exp_index = 0

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

                
                S_i = S.union(set( a[:i+1]))

                local_candidates, num_local_candidates = find_local_candidates(S_i, f, local_X, t)

                num_all_candidates = comm.gather(num_local_candidates, root = 0)
                
                X_i_size = None
                if rank ==0:
                    X_i_size = sum(num_all_candidates)

                X_i_size = comm.bcast(X_i_size, root = 0)
                
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

                
                S_i = S.union(set( a[:i+1]))
                local_candidates, num_local_candidates = find_local_candidates(S_i, f, local_X, t)

                num_all_candidates = comm.gather(num_local_candidates, root = 0)
                
                X_i_size = None
                if rank ==0:
                    X_i_size = sum(num_all_candidates)

                X_i_size = comm.bcast(X_i_size)
                
                if X_i_size < (1 - eps) * len(X):
                    right = i
                else:
                    left = i + 1
                    i = left 

            S.update(a[: i + 1])

            X_i = comm.gather(local_candidates, root = 0)

            if rank == 0:
               X = set()
               for portion in X_i:
                   X.update(portion)

            X = comm.bcast(X, root = 0)
        
        iter = iter + 1

    if rank == 0:
        end_time = MPI.Wtime()
        duration = end_time - start_time
        
        return S, duration
    
    return None, None


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
