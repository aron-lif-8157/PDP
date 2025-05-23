//
// Created by aron on 5/23/25.
//

#ifndef STOICHIOMETRY_HPP
#define STOICHIOMETRY_HPP

#include <array>

constexpr int X_DIM = 7; // different possible stages
constexpr int R = 15;    // different possible outcomes

extern const std::array<std::array<int, X_DIM>, R> P;

#endif // STOICHIOMETRY_HPP
