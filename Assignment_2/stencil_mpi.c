//
// Created by johan on 2025-04-24.
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "stencil.h"

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc != 4) {
		if (rank == 0) fprintf(stderr, "Usage: %s input_file output_file number_of_applications\n", argv[0]);
		MPI_Finalize();
		return 1;
	}
	const char *input_name = argv[1];
	const char *output_name = argv[2];
	int num_steps = atoi(argv[3]);

	int num_values;
	double *input = NULL;
	if (rank == 0) {
		num_values = read_input(input_name, &input);
		if (num_values < 0) {
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
	}
	MPI_Bcast(&num_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int local_n = num_values / size;

	const int STENCIL_WIDTH = 5;
	const int EXTENT = STENCIL_WIDTH / 2;
	double h = 2.0 * PI / num_values;
	const double STENCIL[] = {1.0/(12*h), -8.0/(12*h), 0.0, 8.0/(12*h), -1.0/(12*h)};


	/* Allocate local buffers including halo regions */
	double *local_input = malloc((local_n + 2*EXTENT) * sizeof(double));
	double *local_output = malloc((local_n + 2*EXTENT) * sizeof(double));
	if (!local_input || !local_output) {
		perror("Couldn't allocate local buffers");
		MPI_Abort(MPI_COMM_WORLD, 2);
	}

	/* Scatter data (only core values) */
	MPI_Scatter(input, local_n, MPI_DOUBLE,
				&local_input[EXTENT], local_n, MPI_DOUBLE,
				0, MPI_COMM_WORLD);
	if (rank == 0) free(input);

	int left = (rank - 1 + size) % size;
	int right = (rank + 1) % size;

	/* Synchronize and start timing (exclude I/O and scatter) */
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();

	for (int s = 0; s < num_steps; s++) {
		/* Halo exchange: send core-start to left, recv neighbor-right core-start into right halo */
		MPI_Sendrecv(
			&local_input[EXTENT], EXTENT, MPI_DOUBLE, left, 0,
			&local_input[EXTENT + local_n], EXTENT, MPI_DOUBLE, right, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE
		);
		/* Halo exchange: send core-end to right, recv neighbor-left core-end into left halo */
		MPI_Sendrecv(
			&local_input[EXTENT + local_n - EXTENT], EXTENT, MPI_DOUBLE, right, 1,
			&local_input[0], EXTENT, MPI_DOUBLE, left, 1,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE
		);

		/* Apply stencil on core elements */
		for (int i = EXTENT; i < EXTENT + local_n; i++) {
			double result = 0.0;
			for (int j = 0; j < STENCIL_WIDTH; j++) {
				result += STENCIL[j] * local_input[i - EXTENT + j];
			}
			local_output[i] = result;
		}
		/* Swap buffers for next iteration */
		double *tmp = local_input;
		local_input = local_output;
		local_output = tmp;
	}

	/* Stop timing and compute max across ranks */
	double local_time = MPI_Wtime() - start;
	double max_time;
	MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		printf("%f\n", max_time);
	}

	/* Gather final results (core values) */
#ifdef PRODUCE_OUTPUT_FILE
	double *output = NULL;
	if (rank == 0) {
		output = malloc(num_values * sizeof(double));
		if (!output) MPI_Abort(MPI_COMM_WORLD, 2);
	}
#endif
	MPI_Gather(
		&local_input[EXTENT], local_n, MPI_DOUBLE,
#ifdef PRODUCE_OUTPUT_FILE
		output,
#else
		NULL,
#endif
		local_n, MPI_DOUBLE,
		0, MPI_COMM_WORLD
	);

	if (rank == 0) {
#ifdef PRODUCE_OUTPUT_FILE
		if (write_output(output_name, output, num_values) != 0) {
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		free(output);
#endif
	}

	free(local_input);
	free(local_output);
	MPI_Finalize();
	return 0;
}
