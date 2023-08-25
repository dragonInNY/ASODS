import numpy as np
from mpi4py import MPI
import time

def greedy(f, k, N, S_prev = None):
    '''
        Function:
            Run greedy algorithm on f
        
        Input:
            f: submodular function
            k: integer, cardinality constraint
            N: set, the universe
            S_prev: set, previous solution
    '''

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    if rank == 0:
        start_time = MPI.Wtime()

    if S_prev is not None:
        S = S_prev
    else:
        S = set()

    while len(S) < k:

        # # Method 1: Distribute Data from root
        # if rank == 0:
        #     chunk_size = len(N) // size
        #     list_N = list(N)

        #     for i in range(1, size):
        #         chunk = list_N[(i-1)* chunk_size : i* chunk_size]
        #         comm.send(chunk, dest = i)

        #     recv_data = list_N[(size-1)*chunk_size :]

        # else:

        #     recv_data = comm.recv(source = 0)

        

        # Method 2: Shared Data on every process

        chunk_size = len(N) // size
        remainder = len(N) % size
        list_N = list(N)
        
        if rank < remainder:
            recv_data = list_N[ rank*(chunk_size + 1) : (rank + 1)*(chunk_size+1) ]

        else:
            recv_data = list_N[ rank * chunk_size + remainder: (rank+1) * chunk_size + remainder ]



        # Find elements with greatest marg contribution in each process (local champions)
        local_champion, marg_contribution = find_champion(f, recv_data, S)
        all_champions = comm.gather(local_champion, root = 0)
        all_marg_contribution = comm.gather(marg_contribution, root = 0)


        # #Method 1

        # if rank == 0:

        #     glob_champion = all_champions[np.argmax(all_marg_contribution)]

        #     if glob_champion == None:
        #         break

        #     S.add(glob_champion)
        #     N.remove(glob_champion)

        # S = comm.bcast(S, root = 0)


        # Method 2
        glob_champion = None

        if rank == 0:
            # Find the global champion 
            max_index = np.argmax(all_marg_contribution)
            glob_champion = all_champions[max_index]
            print(glob_champion)
            # Broadcast it to other processes
        
        glob_champion = comm.bcast(glob_champion, root = 0)

        # If no element with positive contribution
        if glob_champion == None:
            break

        S.add(glob_champion)
        N.remove(glob_champion)



    if rank == 0:
        end_time = MPI.Wtime()
        duration = end_time - start_time
        
        return S, duration
    
    return None, None


def find_champion(f, candidates, S):

    '''
        Function:
            Find champion -- 
            the element which has the greatest marginal contribution
        
        Input:
            f: submodular function
            candidates: list, a list of data assigned to the process
            S: set, current solution
    '''

    champion = None
    champion_score = 0

    for candidate in candidates:
        candidate_score = f(S.union({candidate}))
        if candidate_score > champion_score:
            champion = candidate
            champion_score = candidate_score

    return champion, champion_score
