//
// Created by aron on 5/23/25.
//
#include <array>
#include <iostream>
#include <fstream>
#include <string>

#include "ssa.hpp"
#include "stoichiometry.hpp"

// starting values for the state vector
std::array<int, 7> x0{900, 900, 30, 330, 50, 270, 20};

// end time for the simulation
const double T{100.0};

int main(int argc, char *argv[])
{

    //* output file check
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <output_file> <times_to_run>" << std::endl;
        return 1;
    }

    std::ofstream output_file(argv[1]);
    if (!output_file)
    {
        std::cerr << "Error opening output file: " << argv[1] << std::endl;
        return 1;
    }

    int runs{std::stoi(argv[2])}; // Convert second argument to int

    std::random_device rd;     // Random number generator
    //! remeber to change to random seed
    std::mt19937_64 rand_seed(rd()); // Mersenne Twister engine for random number generation
    //! this should be given to the function trough main

    for (int run = 0; run < runs; ++run)
    {
        std::array<int, 7> result{malaria_simulation_sequential(T, x0, rand_seed)};
        for (int i = 0; i < 7; ++i) // Use 7 directly or define X_DIM as 7
        {
            output_file << result[i];
            if (i < 7 - 1)
            {
                output_file << ",";
            }
        }
        output_file << "\n";
    }

    return 0;
}