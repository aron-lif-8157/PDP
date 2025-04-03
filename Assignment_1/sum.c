#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main(int argc, char **argv) {
	if (2 != argc) {
		printf("Usage: sum number_of_summands\n");
		return 1;
	}
	MPI_Init(&argc, &argv);
	// steps är problem size i instruktionerna
	int num_steps = atoi(argv[1]);
	int local_sum = 0;

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	srand(time(NULL) + rank);
	
	double start_time = MPI_Wtime();

	// räkna ut lokal summa
	// 1 << num_steps är 2^num_steps - använder bit shift och flyttar 1 till vänster med num_steps vilket resulterar i 2^num_steps
	for (int i = 0; i < (1 << num_steps); i++) {
		local_sum += rand() % 100;
	}

	// implementera vår logik för parallell summation
	for (int i = 1; i < size; i*=2 ){
		
		// logic for recieving
		if (rank % (2*i)==0) {
			// check if there is a process to receive from
			if (rank + i < size){
				int received;
				// receive the sum from the other process
				MPI_Recv(&received, 1, MPI_INT, rank + i, 0,
						 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				local_sum += received;
			}
		}
		
		// logic for sending
		else {
			int dest = rank - i;
			// send the local sum to the other process
			MPI_Send(&local_sum, 1, MPI_INT, dest, 0,
					 MPI_COMM_WORLD);
			
					 //Process is done and shouldn't continue with the loop
					 break;

		}
	}


	if (rank == 0) {
		printf("Parallel Summation done\n");
		double end_time = MPI_Wtime();
		double elapsed = end_time - start_time;
		printf("%f\n", elapsed);
	}

	// random number generation
	// use command rand() which will give you an integer number


	MPI_Finalize();
	return 0;
}
