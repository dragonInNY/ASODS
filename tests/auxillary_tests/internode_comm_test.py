'''
Test multi-node multi-process
Task: test if inter-node communication takes longer time than
intra-node communication 

'''

from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

node_name = MPI.Get_processor_name()
all_nodes_names = comm.gather(node_name, root=0)

# Identify all nodes
unique_nodes_names = None
if rank == 0:
    unique_nodes_names = sorted(set(all_nodes_names))

unique_nodes_names = comm.bcast(unique_nodes_names, root=0)

# Group processes based on their node
node_index = unique_nodes_names.index(node_name)
subcomm = comm.Split(node_index, rank)
subcomm_rank = subcomm.Get_rank()
subcomm_size = subcomm.Get_size()

# message = "hello from rank {} on node {}".format(subcomm_rank, node_index)
# print(message)

# intra-node communication
start_time = time.time()

for i in range(1000):
    subcomm.send(subcomm_rank, dest = (subcomm_rank +1) % subcomm_size)
    recv_data = subcomm.recv(source=(subcomm_rank - 1) % subcomm_size) 

end_time = time.time()
elapsed_time =  end_time - start_time

# message = "rank {} received from rank {} on node {}".format(subcomm_rank, recv_data, node_index)
# print(message)

total_time = subcomm.reduce(elapsed_time, op = MPI.SUM, root = 0)

if subcomm_rank == 0:
    print('Intra-node communication:', total_time)


# inter-node communication
start_time = time.time()

for i in range(1000):
    comm.send(rank, dest = (rank +24) % size)
    recv_data = comm.recv(source= (rank + 24) % size) 

end_time = time.time()
elapsed_time =  end_time - start_time

# message = "rank {} received from rank {} on node {}".format(rank, recv_data, node_index)
# print(message)

total_time = comm.reduce(elapsed_time, op = MPI.SUM, root = 0)

if rank == 0:
    print('Inter-node communication:', total_time)