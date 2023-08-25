'''
Test featres of MPI

Task: 1. test if this works
single-thread -- multi-thread -- single-thread
2. test if manually initiate/terminate the MPI program will help

Conclusion: once MPI program is initiated, there is no way to
return to single process. 
'''

import mpi4py
mpi4py.rc.initialize = False  # do not initialize MPI automatically
mpi4py.rc.finalize = False # do not finalize MPI automatically

from mpi4py import MPI
import numpy as np


def main_func():
    a = 1
    b = 1
    b = parallel_func(b)

    a = a + b
    # c = parallel_func()

    # a = a+c
    return a


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
    # a = some_func()

    a = 1
    b = 1

    MPI.Init()
    b = parallel_func(b)
    MPI.Finalize()

    a = a + b

    print(a)