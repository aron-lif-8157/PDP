#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>


int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (3 != argc) {
		if (rank == 0) {
		printf("Usage: sum number_of_summands strong/weak (scaling choice)\n");
		}
		MPI_Finalize();
        return 1;
	}

	// steps är problem size i instruktionerna
	double start_time = MPI_Wtime();


	int num_steps = atoi(argv[1]);
	char *scaling_mode = (argv[2]);


	int total_elements, elements_per_process, start, end;

	// check for weak or strong scaling.
	if (strcmp(scaling_mode, "strong") == 0){
		total_elements = 1 << num_steps;
		elements_per_process = total_elements /size;
		int remainder = total_elements % size;
		start = rank * elements_per_process + (rank < remainder ? rank : remainder);
		end = start + elements_per_process + (rank < remainder ? 1 : 0);
	}

	else if (strcmp(scaling_mode, "weak") == 0){
		elements_per_process = 1 << num_steps;
		total_elements = elements_per_process*size;
		start = rank * elements_per_process;
		end = start + elements_per_process;
	}
	else {
		if (rank == 0) printf("Invalid scaling mode use 'strong' or 'weak' as last argument.\n");
		MPI_Finalize();
        return 1;
	}

	// creates unique random number for each thread
	int local_sum = 0;
	srand(time(NULL) + rank);


	// räkna ut lokal summa
	// 1 << num_steps är 2^num_steps - använder bit shift och flyttar 1 till vänster med num_steps vilket resulterar i 2^num_steps
    for (int i = start; i < end; i++) { 
        local_sum += rand() % 100;
    }
		
	int active = 1;
	while (active < size) {
		if (rank % (2 * active) == 0) {
			if (rank + active < size) {
				int received;
				MPI_Recv(&received, 1, MPI_INT, rank + active, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				local_sum += received;
			}
		} else {
			int dest = rank - active;
			MPI_Send(&local_sum, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
			break;
		}
		active *= 2;
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
