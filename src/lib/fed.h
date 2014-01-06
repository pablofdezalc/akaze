/**
 * @file fed.h
 * @brief Functions for performing Fast Explicit Diffusion and building the
 * nonlinear scale space
 * @date Sep 15, 2013
 * @author Pablo F. Alcantarilla, Jesus Nuevo
 * @note This code is derived from FED/FJ library from Grewenig et al.,
 * The FED/FJ library allows solving more advanced problems
 * Please look at the following papers for more information about FED:
 * [1] S. Grewenig, J. Weickert, C. Schroers, A. Bruhn. Cyclic Schemes for
 * PDE-Based Image Analysis. Technical Report No. 327, Department of Mathematics,
 * Saarland University, Saarbrücken, Germany, March 2013
 * [2] S. Grewenig, J. Weickert, A. Bruhn. From box filtering to fast explicit diffusion.
 * DAGM, 2010
 *
*/

#pragma once

/* ************************************************************************* */

// Includes
#include <iostream>
#include <vector>

/* ************************************************************************* */
// Declaration of functions
int fed_tau_by_process_time(const float& T, const int& M, const float& tau_max,
                            const bool& reordering, std::vector<float>& tau);
int fed_tau_by_cycle_time(const float& t, const float& tau_max,
                          const bool& reordering, std::vector<float> &tau) ;
int fed_tau_internal(const int& n, const float& scale, const float& tau_max,
                     const bool& reordering, std::vector<float> &tau);
bool fed_is_prime_internal(const int& number);

/* ************************************************************************* */

