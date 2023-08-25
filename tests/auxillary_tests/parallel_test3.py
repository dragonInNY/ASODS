'''
Test featres of MPI

Task: test if calling a function which uses MPI with 
different parameters will change the data in the processes

Conclusion: it will
'''

from mpi4py import MPI
import numpy as np

def main_func():
    b = 1
    b = parallel_func(b)

    b = 3
    b = parallel_func(b)

    return b


def parallel_func(b):

    # MPI.Init()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    b_s = comm.gather(b, root = 0)

    if rank == 0:
        b = np.sum(b_s)
    
    comm.Barrier()

    # MPI.Finalize()

    print('rank', rank, 'b', b)

    return b

if __name__ == "__main__":

    a = main_func()
