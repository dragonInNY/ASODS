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

        # keep track of the newest f(S)
        latest_f_S = 0

    if S_prev is not None:
        S = S_prev
    else:
        S = set()


    while len(S) < k:

        if rank == 0:
            old_champ_score = latest_f_S
            lazy_search_sentinal = size

        chunk_size = len(N) // size
        remainder = len(N) % size
        list_N = list(N)
        
        if rank < remainder:
            recv_data = list_N[ rank*(chunk_size + 1) : (rank + 1)*(chunk_size+1) ]

        else:
            recv_data = list_N[ rank * chunk_size + remainder: (rank+1) * chunk_size + remainder ]

        # Find elements with greatest marg contribution in each process (local champions)
        local_candidate_score = evaluate_local_candidates(f, recv_data, S)
        all_candidate_score = comm.gather(local_candidate_score, root = 0)

        # release memory
        del local_candidate_score

        glob_champ = None
        if rank == 0:
            all_candidate_score = np.concatenate(all_candidate_score)

            # sort all scores and rearrange list_N accordingly
            sorted_indices = sorted(range(len(all_candidate_score)), key = lambda x: all_candidate_score[x], reverse = True)

            all_candidate_score_sorted = [all_candidate_score[i] for i in sorted_indices]
            list_N_sorted = [list_N[i] for i in sorted_indices]

            # find global champion
            if all_candidate_score_sorted[0] <= latest_f_S:
                glob_champ = None 
            else:
                glob_champ = list_N_sorted[0]

                #update everything
                latest_f_S = all_candidate_score_sorted[0]
                list_N_sorted.pop(0)
                all_candidate_score_sorted.pop(0)

        # release memory
        del list_N 
        
        glob_champ = comm.bcast(glob_champ, root = 0)

        # If no element with positive marginal contribution, terminate
        if glob_champ == None:
            break

        S.add(glob_champ)
        N.remove(glob_champ)

        # Lazy greedy

        lazy_search_domain = None
        if rank == 0:

            # the (size+1)-th score should be the threshold
            threshold = all_candidate_score_sorted[lazy_search_sentinal] - old_champ_score
            lazy_search_domain = list_N_sorted[: lazy_search_sentinal ]

            # the champ before expanding lazy search domain
            old_lazy_champ = None
            old_lazy_champ_score = 0


        while len(S) < k:

            lazy_search_domain = comm.bcast(lazy_search_domain, root = 0)
            chunk_size = len(lazy_search_domain) // size
            assigned_champ = lazy_search_domain[rank * chunk_size: (rank+1)* chunk_size]
            upd_champ_score = evaluate_local_candidates(f, assigned_champ, S)

            all_upd_champ_score = comm.gather(upd_champ_score, root = 0)

            # release memory
            del assigned_champ
            del upd_champ_score

            # start a new complete round
            START_NEW_ROUND = False

            if rank == 0:
                all_upd_champ_score = np.concatenate(all_upd_champ_score)
                max_index = np.argmax(all_upd_champ_score)

                # champ in this lazy search domain
                current_lazy_champ = lazy_search_domain[max_index]
                current_lazy_champ_score = all_upd_champ_score[max_index]
                
                # update the champ in the range before sentinel
                if current_lazy_champ_score < old_lazy_champ_score:
                    current_lazy_champ = old_lazy_champ
                    current_lazy_champ_score = old_lazy_champ_score

                # if we find a glob_champ
                if current_lazy_champ_score - latest_f_S >= threshold:
                    

                    glob_champ = current_lazy_champ

                    #update everything
                    latest_f_S = current_lazy_champ_score
                    list_N_sorted.remove(glob_champ)
                    all_candidate_score_sorted.pop(0)

                    lazy_search_domain = list_N_sorted[: lazy_search_sentinal]
                    threshold = all_candidate_score_sorted[lazy_search_sentinal] - old_champ_score
                    

                # if not, expand lazy search domain
                else:
                    glob_champ = None

                    lazy_search_sentinal = lazy_search_sentinal + size
                    lazy_search_domain = list_N_sorted[ lazy_search_sentinal - size: lazy_search_sentinal]
                    threshold = all_candidate_score_sorted[lazy_search_sentinal] - old_champ_score

                    old_lazy_champ = current_lazy_champ
                    old_lazy_champ_score = current_lazy_champ_score

                # if sentinel is out of range, start a new round
                if lazy_search_sentinal >= len(list_N_sorted):
                        START_NEW_ROUND = True
            
            START_NEW_ROUND = comm.bcast(START_NEW_ROUND, root = 0)
            if START_NEW_ROUND:
                break
            
            glob_champ = comm.bcast(glob_champ, root = 0)
            if glob_champ == None:
                continue

            S.add(glob_champ)
            N.remove(glob_champ)

    if rank == 0:
        end_time = MPI.Wtime()
        duration = end_time - start_time
        
        return S, duration
    
    return None, None


def evaluate_local_candidates(f, candidates, S):

    '''
        Function:
            Evaluate all local candidates
        
        Input:
            f: submodular function
            candidates: list, a list of data assigned to the process
            S: set, current solution
    '''

    candidates_score = [ f(S.union({candidate})) for candidate in candidates ]

    return candidates_score
