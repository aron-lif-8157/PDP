#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double start_time = MPI_Wtime();
	int d = 1;
	int local_sum = 0;
	//std::cout << "Size" << (1 << 22) << std::endl;
	for (int i = 0; i < (1 << 19); i++) {
		local_sum = rand() % 100 + local_sum;
	}
	while (d < size) {
		if (rank % (2 * d) == 0) {
			if (rank + d < size) {
				int received;
				MPI_Recv(&received, 1, MPI_INT, rank + d, 0,
						 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				local_sum += received;
			}
		} else {
			int dest = rank - d;
			MPI_Send(&local_sum, 1, MPI_INT, dest, 0,
					 MPI_COMM_WORLD);
			break;
		}
		d *= 2;
	}

	if (rank == 0) {
//std::cout << "Global sum is: " << local_sum << std::endl;
	}


	double end_time = MPI_Wtime();
	double elapsed = end_time - start_time;
	printf("%f\n", elapsed);
	//std::cout << "Time:" << elapsed << std::endl;
	MPI_Finalize();

	return 0;
}