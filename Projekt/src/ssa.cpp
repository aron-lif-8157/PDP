//
// Created by aron on 5/23/25.
//
#include <array>
#include <random>

#include "ssa.hpp"
#include "stoichiometry.hpp"

extern "C"
{
    void prop(int *x, double *w);
}
const double T{100};

/*
* @brief Sequential simulation of the Malaria model using the Stochastic Simulation Algorithm (SSA).
* @param T The end time for the simulation.
* @param x0 The initial state vector of the system, represented as an array of integers.
* @return The final state vector after the simulation, represented as an array of integers.
*/
std::array<int, 7> malaria_simulation_sequential(double T, std::array<int, 7> x0, std::mt19937_64 &ran_seed)
{

    double t{0};                                          // Initial time
    std::uniform_real_distribution<double> dis(0.0, 1.0); // Uniform distribution for random numbers
    double random_time_increment{0.0};                    // Variable to hold the random time increment
    std::array<double, 15> w{0};                          // initial w value

    while (t < T) //! step 2

    {
        double a0 = 0; // initial value of a0

        //! step 3
        prop(x0.data(), w.data());

        //! step 4
        //? Can be parallelized
        for (int i = 0; i < R; ++i)
        {
            // calculate w0
            a0 += w[i];
        }
        if (a0 <= 0)
            break;
        // If a0 is zero or negative, exit the loop
        // can't divide by 0

        //! step 5
        double u1{dis(ran_seed)}; // Generate a random number between 0 and 1
        double u2{dis(ran_seed)}; // Generate another random number between 0 and 1

        //! step 6
        random_time_increment = -std::log(u1) / a0; // Calculate the random time increment

        //! step 7
        double threshold {u2 * a0};
        double cumulative_sum {0.0}; // Initialize cumulative sum
        int r {0};                   // Reaction index
        for (; r < R; ++r)
        {
            cumulative_sum += w[r]; // Update cumulative sum
            if (cumulative_sum >= threshold)
                break;
        }

        //! step 8
        for (int i = 0; i < X_DIM; ++i)
        {
            x0[i] += P[r][i]; // Update the state vector x0 based on the selected reaction
        }

        //! step 9
        t += random_time_increment; // Update the time by adding the random time increment
    }
    return x0; // Return the final state vector after the simulation    
}