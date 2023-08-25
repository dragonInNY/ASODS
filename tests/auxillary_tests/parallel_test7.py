'''
    Test if comm.gather gathers data sequentially

    Conclusion: it does!
'''

from mpi4py import MPI
import numpy as np
import sys

def main_func():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ranks = comm.gather( rank, root = 0)

    if rank == 0:
        print(ranks)

    return None


if __name__ == "__main__":

    a = main_func()