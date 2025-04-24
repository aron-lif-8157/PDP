#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "stencil.h"

#define STENCIL_WIDTH 5
#define EXTENT (STENCIL_WIDTH/2)

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	if (argc != 4) {
		if (rank == 0) {
			fprintf(stderr, "Usage: stencil input_file output_file number_of_applications\n");
		}
		MPI_Finalize();
		return 1;
	}
	char *input_name = argv[1];
	char *output_name = argv[2];
	int num_steps = atoi(argv[3]);

	int num_values = 0;
	double *input = NULL;

	if (rank == 0) {
		num_values = read_input(input_name, &input);
		if (num_values < 0) {
			MPI_Abort(comm, 2);
		}
	}
	MPI_Bcast(&num_values, 1, MPI_INT, 0, comm);

	int q = num_values / size;
	int rem = num_values % size;
	int *sendcounts = malloc(size * sizeof(int));
	int *displs     = malloc(size * sizeof(int));
	for (int i = 0; i < size; i++) {
		sendcounts[i] = q + (i < rem ? 1 : 0);
		displs[i]     = (i == 0 ? 0 : displs[i-1] + sendcounts[i-1]);
	}
	int local_n = sendcounts[rank];

	double *local_data = malloc(local_n * sizeof(double));
	MPI_Scatterv(input, sendcounts, displs, MPI_DOUBLE,
				 local_data, local_n, MPI_DOUBLE,
				 0, comm);
	if (rank == 0) free(input);

	double h = 2.0 * PI / num_values;
	double stencil[STENCIL_WIDTH] = {
		1.0/(12.0*h),
		-8.0/(12.0*h),
		0.0,
		8.0/(12.0*h),
		-1.0/(12.0*h)
	};

	int buf_size = local_n + 2 * EXTENT;
	double *old = malloc(buf_size * sizeof(double));
	double *new = malloc(buf_size * sizeof(double));
	for (int i = 0; i < EXTENT; i++)      old[i]                 = 0.0;
	for (int i = 0; i < local_n;  i++)      old[EXTENT + i]       = local_data[i];
	for (int i = 0; i < EXTENT; i++)      old[EXTENT+local_n + i] = 0.0;
	free(local_data);

	int left  = (rank - 1 + size) % size;
	int right = (rank + 1)       % size;

	MPI_Barrier(comm);
	double start = MPI_Wtime();
	for (int step = 0; step < num_steps; step++) {
		// Halo exchange
		MPI_Sendrecv(&old[EXTENT], EXTENT, MPI_DOUBLE, left,  0,
					 &old[EXTENT + local_n], EXTENT, MPI_DOUBLE, right, 0,
					 comm, MPI_STATUS_IGNORE);
		MPI_Sendrecv(&old[EXTENT + local_n - EXTENT], EXTENT, MPI_DOUBLE, right, 1,
					 &old[0],                 EXTENT, MPI_DOUBLE, left,  1,
					 comm, MPI_STATUS_IGNORE);
		// Compute stencil
		for (int i = EXTENT; i < EXTENT + local_n; i++) {
			double sum = 0.0;
			for (int j = 0; j < STENCIL_WIDTH; j++) {
				sum += stencil[j] * old[i - EXTENT + j];
			}
			new[i] = sum;
		}
		double *tmp = old; old = new; new = tmp;
	}
	MPI_Barrier(comm);

	double local_time;
	local_time = MPI_Wtime() - start;
	double max_time;
	MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
	if (rank == 0) {
		printf("%f\n", max_time);
	}

	double *final_output = NULL;
	if (rank == 0) final_output = malloc(num_values * sizeof(double));
	MPI_Gatherv(&old[EXTENT], local_n, MPI_DOUBLE,
				final_output, sendcounts, displs, MPI_DOUBLE,
				0, comm);

	if (rank == 0) {
		if (0 != write_output(output_name, final_output, num_values)) {
			MPI_Abort(comm, 2);
		}
	}

	free(old);
	free(new);
	free(sendcounts);
	free(displs);
	if (rank == 0) free(final_output);

	MPI_Finalize();
	return 0;
}

int read_input(const char *file_name, double **values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "r"))) {
		perror("Couldn't open input file");
		return -1;
	}
	int num_values;
	if (EOF == fscanf(file, "%d", &num_values)) {
		perror("Couldn't read element count from input file");
		return -1;
	}
	if (NULL == (*values = malloc(num_values * sizeof(double)))) {
		perror("Couldn't allocate memory for input");
		return -1;
	}
	for (int i = 0; i < num_values; i++) {
		if (EOF == fscanf(file, "%lf", &((*values)[i]))) {
			perror("Couldn't read elements from input file");
			return -1;
		}
	}
	if (0 != fclose(file)) {
		perror("Warning: couldn't close input file");
	}
	return num_values;
}

int write_output(char *file_name, const double *output, int num_values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "w"))) {
		perror("Couldn't open output file");
		return -1;
	}
	for (int i = 0; i < num_values; i++) {
		if (0 > fprintf(file, "%.4f ", output[i])) {
			perror("Couldn't write to output file");
		}
	}
	if (0 > fprintf(file, "\n")) {
		perror("Couldn't write to output file");
	}
	if (0 != fclose(file)) {
		perror("Warning: couldn't close output file");
	}
	return 0;
}
