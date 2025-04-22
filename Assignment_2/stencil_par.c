#include "stencil.h"
#include <mpi.h>
#include <time.h>

int main(int argc, char **argv)
{
    // vi initierar processerna
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Process 0 checks for arguments
    if (4 != argc)
    {
        if (rank == 0)
        {
            printf("Usage: stencil input_file output_file number_of_applications\n");
        }
        MPI_Finalize();
        return 1;
    }

    char *input_name = argv[1];
    char *output_name = argv[2];
    int stencil_count = atoi(argv[3]);
    int num_values;

    /*
     *   A function that reads the input file
     *   divides the work to other processes.
     */
    if (rank == 0)
    {
        // Read input file
        double *input;

        // check if the input file could be read
        if (0 > (num_values = read_input(input_name, &input)))
        {
            printf("Error reading input file\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        // check if the number of values is divisible by the number of processes
        if (num_values % size != 0)
        {
            printf("Number of values is not divisible by number of processes\n");
            printf("Number of values needs to be divisible by number of processes as per the assignment intructions\n");
            printf("Number of values: %d, number of processes: %d\n", num_values, size);
            printf("please change process count\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }

    //! Stencil values
    // TODO: Detta kanske behöver allokeras i heap minne
    double h = 2.0 * PI / num_values;
    const int STENCIL_WIDTH = 5;
    const int EXTENT = STENCIL_WIDTH / 2;
    const double STENCIL[] = {1.0/(12*h), -8.0/(12*h), 0.0, 8.0/(12*h), -1.0/(12*h)};


    /*  We broadcast number of steps to all the recivers
        TODO: there might be a problem since num_values isn't heap allocated
     *   Broadcasting and receving in one function call
     *   &num_values, address of value to be send and address where to store recived value
     *   1, number of values to be send
     *   MPI_INT, type of value to be send
     *   0, rank of process that sends the value
     *   MPI_COMM_WORLD, communicator to be used
     */

    MPI_Bcast(&num_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int num_steps = num_values / size;
    // TODO : num_steps kanske borde vara heap allokerad

    // Allocate data for result including halo values
    double *local_recived = mallock((num_steps + 2 * EXTENT) * sizeof(double));
    double *local_result = mallock((num_steps) * sizeof(double));

    /*  MPI Scatter Scatter the input data to all processes
     ?   input, pointer to the data to be scattered
     ?   num_steps, number of values to be scattered
     ?   MPI_DOUBLE, type of data to be scattered
     ?   local_recived, pointer to where the data should be recived
     ?   num_steps, number of values to be recived
     ?   MPI_DOUBLE, type of data to be recived
     ?   0, rank of process that sends the data
     ?   MPI_COMM_WORLD, communicator to be used
     */

    // TODO: Varför scatterar vi num_steps när vi redan har räknat ut det i varje process
    MPI_Scatter(input, // maybe should be pointer
                num_steps,
                MPI_DOUBLE,
                local_recived[EXTENT],
                num_steps,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD);

    //! Apply stencil with same loop logic as seen in stencil.c
    for (int count = 0; count < stencil_count; count++)
    {
        int left_neighbour = (rank == 0 ? size - 1 : rank - 1);
        int right_neighbour = (rank == size - 1 ? 0 : rank + 1);
        // Handle halo transfeer between processes.

        //! Send left halo to left process
        MPI_Sendrecv(local_recived[EXTENT + num_steps], // address to first element that is to be sent (the right most element)
                     EXTENT,                            // number of elements to be sent
                     MPI_DOUBLE,                        // type of data to be sent
                     left_neighbour,                    // Destination rank for the data
                     0,                                 // tag for the data needed for the send

                     &local_recived[0] // address to where the data should be recived
                     EXTENT,           // number of elements to be recived
                     MPI_DOUBLE,       // type of data to be recived
                     right_neighbour,  // rank of process that sends the data
                     0,                // tag for the data needed for the recive

                     MPI_COMM_WORLD,   // communicator to be used
                     MPI_STATUS_IGNORE // status of the recive
        );

        //! Send right halo to right process
        MPI_Sendrecv(&local_recived[EXTENT], // address to first element that is to be sent (the left most element)
                     EXTENT,                 // number of elements to be sent
                     MPI_DOUBLE,             // type of data to be sent
                     right_neighbour,        // Destination rank for the data
                     0,                      // tag for the data needed for the send

                     &local_recived[num_steps + EXTENT], // address to where the data should be recived
                     EXTENT,                             // number of elements to be recived
                     MPI_DOUBLE,                         // type of data to be recived
                     left_neighbour,                     // rank of process that sends the data
                     0,                                  // tag for the data needed for the recive

                     MPI_COMM_WORLD,   // communicator to be used
                     MPI_STATUS_IGNORE // status of the recive
        );

        //! applying to each element
        for (int s = 0; s < num_steps; s++)
        {

            for (int i = EXTENT; i < num_steps + EXTENT; i++)
            {
                double result = 0;
                for (int j = 0; j < STENCIL_WIDTH; j++)
                {
                    int index = (i - EXTENT + j) % num_values;
                    result += STENCIL[j] * local_recived[index];
                }
                local_result[i - EXTENT] = result;
            }
        }
        //! update the local_recived with the result
        for (int i = 0; i < num_steps; i++)
        {
            local_recived[i + EXTENT] = local_result[i];
        }
    }
    int double *output = NULL;
    if (NULL == (output = malloc(num_values * sizeof(double)))) {
		perror("Couldn't allocate memory for output");
		return 2;
	}
    //! gather result
    MPI_Gather(local_result[0], // address to first element that is to be sent (the left most element)
                num_steps,             // number of elements to be sent
                MPI_DOUBLE,            // type of data to be sent
                output,                // address to where the data should be recived
                num_steps,             // number of elements to be recived
                MPI_DOUBLE,            // type of data to be recived
                0,                     // rank of process that sends the data
                MPI_COMM_WORLD         // communicator to be used
    );
}