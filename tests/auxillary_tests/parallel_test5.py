'''
    Test when comm is passed into a function as a parameter
    data will be changed by the function

    Conclusion: it will not
'''

from mpi4py import MPI
import numpy as np
import sys

def main_func():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data = 1
    
    gathered_data =parallel_func(comm, data)

    # print('checkpoint4')
    
    print('rank', rank, 'data', data, gathered_data)

    return gathered_data


def parallel_func(comm, data):

    # print('checkpoint2')

    # print(data)

    data = comm.Get_rank()

    gathered_data = comm.gather(data, root = 0)

    # print('checkpoint3')

    return gathered_data

if __name__ == "__main__":

    a = main_func()