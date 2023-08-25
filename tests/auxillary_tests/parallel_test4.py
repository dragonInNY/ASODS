'''
Test featres of MPI

Task: test if calling COMM_WORLD twice will obliterate
the data stored at the first time

Conclusion: should pass comm as a parameter to another function.
Be aware that comm.bcast is a collective operation
'''

from mpi4py import MPI
import numpy as np
import sys

def main_func():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data = None

    if rank == 0:
        data = 1
        
    data = comm.bcast(data, root = 0)

    comm.Barrier()

    # print('checkpoint1')
    
    gathered_data =parallel_func(comm, data)

    # print('checkpoint4')
    
    print('rank', rank, 'data', data)

    return gathered_data


def parallel_func(comm, data):

    # print('checkpoint2')

    # print(data)

    gathered_data = comm.gather(data, root = 0)

    # print('checkpoint3')

    return gathered_data

if __name__ == "__main__":

    a = main_func()