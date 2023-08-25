'''
Test if parallel computing works
Task: i-th process sends i to (i+1)-th process

'''

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Start the timer
if rank == 0:
    start_time = MPI.Wtime()

# Calculate the ranks of the sending and receiving processes
send_rank = (rank + 1) % size
recv_rank = (rank - 1) % size

# Send the rank to the next process
comm.send(rank, dest=send_rank)

# Receive the rank from the previous process
received_rank = comm.recv(source=recv_rank)

# Print the received rank
# print("Process", rank, "received rank", received_rank)

node_name = MPI.Get_processor_name()
print(node_name)

# Stop the timer 
if rank == 0:
    end_time = MPI.Wtime()
    execution_time = end_time - start_time
    print("Total execution time:", execution_time, "seconds")