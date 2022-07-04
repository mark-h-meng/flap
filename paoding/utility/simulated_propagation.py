#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

import paoding.utility.interval_arithmetic as ia
import paoding.utility.utils as utils
import math


def calculate_bounds_of_output(model, intervals, loc):
    # Load the parameters and configuration of the input model
    (w, g) = utils.load_param_and_config(model)

    num_layers = len(model.layers)

    # Just return these intervals if current location is at the 2nd last layer
    if loc == num_layers - 1:
        return intervals

    total_pruned_count = 0

    propagated_next_layer_interval = None

    while loc < num_layers - 1:
        # Exclude non FC layers
        num_curr_neurons = len(w[loc + 1][0])
        num_next_neurons = len(w[loc + 1][0][0])

        relu_activation = g[loc]['activation'] == 'relu'

        if len(intervals) != num_curr_neurons:
            raise Exception("Error: input intervals are not in expected shape -",
                            num_curr_neurons, "expected, not", len(intervals))

        # No activation at the output layer
        if loc + 1 == num_layers - 1:
            propagated_next_layer_interval = ia.forward_propogation(intervals,
                                                                    w[loc + 1][0],
                                                                    w[loc + 1][1],
                                                                    activation=False)
        else:
            propagated_next_layer_interval = ia.forward_propogation(intervals,
                                                                    w[loc + 1][0],
                                                                    w[loc + 1][1],
                                                                    activation=True,
                                                                    relu_activation=relu_activation)
        intervals = propagated_next_layer_interval
        loc += 1

    return propagated_next_layer_interval


# Return the evaluation of the impact in a pair of real numbers as interval
def calculate_impact_of_pruning_next_layer(model, big_map, pruning_pairs, loc, cumulative_next_layer_intervals=None,
                                           kaggle_credit=False):
    # Load the parameters and configuration of the input model
    (w, g) = utils.load_param_and_config(model)

    # Each pruning pair is in form of a tuple (a,b), in which "a" is the hidden unit to be pruned, and "b"
    #   is the one to remain. The Delta produced by this pruning is as follow:
    #                 Delta = [b * (w_a + w_b) + 2 * bias_b] - [a * w_a + bias_a + b * w_b + bias_b]
    #                       = (b-a) * w_a + (bias_b - bias_a)
    #   or if we omit the impact of bias:
    #                 Delta = [b * (w_a + w_b)] - [a * w_a + b * w_b]
    #                       = (b-a) * w_a
    # The Delta produced by each pruning is presented at the next layer, and the propagation
    #   simulates the impact of Delta to the output layer

    # In case there is a single unit pruning, s.t. b = -1
    #    the Delta will be -1 * (a * w_a)

    next_layer_size = len(w[loc+1][0][0])
    if cumulative_next_layer_intervals is None:
        empty_interval = (0,0)
        cumulative_next_layer_intervals = [empty_interval for i in range(0, next_layer_size)]

    num_layers = len(model.layers)
    for (a, b) in pruning_pairs:
        (a_lo, a_hi) = big_map[loc][a]
        # DEPRECATED
        # (a_lo, a_hi) = get_definition_interval(a, loc, parameters=w, relu_activation=use_relu, kaggle_credit=kaggle_credit)

        # Check if there is a pair pruning or single unit pruning (b=-1)
        if b != -1:
            (b_lo, b_hi) = big_map[loc][b]
            # DEPRECATED
            # (b_lo, b_hi) = get_definition_interval(b, loc, parameters=w, relu_activation=use_relu, kaggle_credit=kaggle_credit)

            # approximate the result of (a-b)
            (a_minus_b_lo, a_minus_b_hi) = ia.interval_minus((a_lo, a_hi), (b_lo, b_hi))
            w_a = w[loc + 1][0][a]
            if len(w_a) is not next_layer_size:
                raise Exception("Inconsistent size of parameters")

            impact_to_next_layer = [ia.interval_scale((a_minus_b_lo, a_minus_b_hi), k) for k in w_a]
        else:
            w_a = w[loc + 1][0][a]
            if len(w_a) is not next_layer_size:
                raise Exception("Inconsistent size of parameters")

            impact_to_next_layer = [ia.interval_scale((a_lo, a_hi), -1*k) for k in w_a]

        if len(impact_to_next_layer) is not next_layer_size:
            raise Exception("Inconsistent size of parameters")

        for index, interval in enumerate(cumulative_next_layer_intervals):
            cumulative_next_layer_intervals[index] = ia.interval_add(interval, impact_to_next_layer[index])

        #print(cumulative_next_layer_intervals)

    return cumulative_next_layer_intervals


def get_definition_map(model, definition_dict=None, input_interval=(0, 1)):

    # First locate the dense (FC) layers, starting from the input layer/flatten layer until the second last layer
    ## Load the parameters and configuration of the input model
    (w, g) = utils.load_param_and_config(model)
    num_layers = len(model.layers)
    layer_idx = 0
    starting_layer_index = -1
    ending_layer_index = -1
    while layer_idx < num_layers - 1:
        if "dense" in model.layers[layer_idx].name:
            if starting_layer_index < 0:
                starting_layer_index = layer_idx - 1
            if ending_layer_index < layer_idx:
                ending_layer_index = layer_idx
        layer_idx += 1

    if (starting_layer_index < 0) or (ending_layer_index < 0):
        raise Exception("Fully connected layers not identified")

    # Now let's create a hash table as dictionary to store all definition intervals of FC neurons
    if definition_dict is None:
        definition_dict = {}
        definition_dict[starting_layer_index] = {}

    for i in range(0, len(w[starting_layer_index + 1][0])):
        definition_dict[starting_layer_index][i] = input_interval

    for i in range(starting_layer_index + 1, ending_layer_index + 1):
        num_prev_neurons = len(w[i][0])
        num_curr_neurons = len(w[i][0][0])
        if i not in definition_dict.keys():
            definition_dict[i] = {}

        curr_activation = g[i]['activation']

        for m in range(0, num_curr_neurons):
            (sum_lo, sum_hi) = (0, 0)
            for n in range(0, num_prev_neurons):
                affine_w_x = ia.interval_scale(definition_dict[i-1][n], w[i][0][n][m])
                (sum_lo, sum_hi) = ia.interval_add((sum_lo, sum_hi), affine_w_x)
            bias = (w[i][1][m], w[i][1][m])
            (sum_lo, sum_hi) = ia.interval_add((sum_lo, sum_hi), bias)

            if curr_activation == 'relu':
                definition_dict[i][m] = (0, sum_hi)
            else: # Assume it is sigmoid
                sum_hi =  1 / (1 + math.exp(-1 * sum_hi))
                sum_lo =  1 / (1 + math.exp(-1 * sum_lo))
                definition_dict[i][m] = (sum_lo, sum_hi)

    return definition_dict


# DEPRECATED - Replaced by initialize_definition_map
def get_definition_interval(unit_index, layer_index, parameters, relu_activation=True, kaggle_credit=False):

    if kaggle_credit:
        input_definition_interval = (-5, 5)
    else:
        input_definition_interval = (0, 1)
    # input_size = len(parameters[1][0])
    # Starting from input layer (MLP) or the last flatten layer (CNN)
    if layer_index == 1 or (layer_index>1 and not parameters[layer_index-1]):
        #print(">> DEBUG: unit_index:", unit_index, " & layer_index:", layer_index)
        weights = [parameters[layer_index][0][j][unit_index] for j in range(0, len(parameters[layer_index][0]))]
        bias = parameters[layer_index][1][unit_index]
        (sum_lo, sum_hi) = ia.interval_sum([ia.interval_scale(input_definition_interval, w) for w in weights])
        (sum_lo, sum_hi) = ia.interval_add((sum_lo, sum_hi), (bias, bias))
        if relu_activation:
            if sum_hi < 0:
                sum_hi = 0
            if sum_lo < 0:
                sum_lo = 0
        else:
            sum_hi =  1 / (1 + math.exp(-1 * sum_hi))
            sum_lo =  1 / (1 + math.exp(-1 * sum_lo))
        return (sum_lo, sum_hi)

    # Temp Wordaround: no definition algorithm avaliable for nodes after the 2nd layer, set as [-1,1]
    else:
        weights = [parameters[layer_index][0][j][unit_index] for j in range(0, len(parameters[layer_index][0]))]
        bias = parameters[layer_index][1][unit_index]
        (sum_lo, sum_hi) = ia.interval_sum([ia.interval_scale(input_definition_interval, w) for w in weights])
        (sum_lo, sum_hi) = ia.interval_add((sum_lo, sum_hi), (bias, bias))
        if relu_activation:
            if sum_hi < 0:
                sum_hi = 0
            if sum_lo < 0:
                sum_lo = 0
        else:
            sum_hi = 1 / (1 + math.exp(-1 * sum_hi))
            sum_lo = 1 / (1 + math.exp(-1 * sum_lo))
        return (sum_lo, sum_hi)
    return None
