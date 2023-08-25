'''
    Test if I can use func_with_comm inside rank == 0

    Conclusion: it will only use rank0 process
'''

from mpi4py import MPI
import numpy as np
import sys

def main_func():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data = 1
    gathered_data = None
    
    if rank == 0:

        gathered_data = parallel_func(comm, data)
    

    return gathered_data


def parallel_func(comm, data):

    # print('checkpoint2')

    # print(data)

    rank = comm.Get_rank()

    print(rank)

    # print('checkpoint3')

    return rank

if __name__ == "__main__":

    a = main_func()