#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

import numpy as np
import paoding.utility.interval_arithmetic as ia

def build_saliency_matrix(curr_parameters, next_paramters):
    size = len(curr_parameters)
    matrix = np.zeros((size, size))
    for i, x in enumerate(curr_parameters):
        for j, y in enumerate(curr_parameters):
            # The saliency is calculated as s_(i,j)=(a_j)^2 * ||e_(i,j)||^2
            matrix[i, j] = np.linalg.norm(x-y) * np.linalg.norm(next_paramters[j])
    return matrix

def build_saliency_matrix_with_bias(curr_parameters, next_paramters, bias_parameters):
    size = len(curr_parameters)
    matrix = np.zeros((size, size))
    for i, x in enumerate(curr_parameters):
        for j, y in enumerate(curr_parameters):
            # The saliency is calculated as s_(i,j)=(a_j) * (||e_(i,j)||+|b_i-b_j|/|b_i+b_j|)
            bias_term = (bias_parameters[i] - bias_parameters[j]) / (bias_parameters[i] + bias_parameters[j])
            matrix[i, j] = (abs(bias_term) + np.linalg.norm(x-y)) * np.linalg.norm(next_paramters[j])
    return matrix


def build_impact_propagation_matrix(curr_parameters, next_parameters, prev_units_bounds):
    (prev_units_bounds_lo, prev_units_bounds_hi) = prev_units_bounds
    if not len(prev_units_bounds_lo)==len(curr_parameters[0]) or not len(prev_units_bounds_lo)==len(prev_units_bounds_hi):
        return None
    #size = len(curr_parameters)
    #matrix = np.zeros((size, size))
    matrix = []
    # To interpret the matrix, the row corr. unit is the one to be pruned, and the column corr. units
    #   are the units remaining.
    for i, x in enumerate(curr_parameters):
        matrix_row = []
        for j, y in enumerate(curr_parameters):
            # The DELTA is in form of: (w1 + w2)*a2 - (w1*a1 + w2*a2) = w1*(a2 - a1)
            #   where w1 is a vector of all weight params. connecting to the next layer from a1
            w1 = next_parameters[i]
            #   and diff_unit is the result of (a2 - a1)
            diff_unit = ia.interval_minus((prev_units_bounds_lo[j], prev_units_bounds_hi[j]), (prev_units_bounds_lo[i], prev_units_bounds_hi[i]))
            propagated_interval_to_next_layer = [ia.interval_scale(diff_unit, w) for w in w1]
            # Here we return a list of interval for each item within the matrix
            matrix_row.append(propagated_interval_to_next_layer)
        matrix.append(matrix_row)
    return matrix

