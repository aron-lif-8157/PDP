//
// Created by aron on 5/23/25.
//
#include <array>

#include "ssa.hpp"
#include "stoichiometry.hpp"

extern "C" {
    void prop(int *x, double *w);
}
double t {0};
const double T {100};
std::array<int,7> x0 {900,900,30,330,50,270,20};
std::array<double,15> w {0};


int malaria_simulation_sequential (double t, double T, std::array<int,7> x0) {
    std::array<int,7> w {0};

    while (t < T) {
    w = 
    }
    return 0;
}