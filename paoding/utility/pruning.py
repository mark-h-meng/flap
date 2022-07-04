#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

import paoding.utility.bcolors as bcolors
import paoding.utility.saliency as saliency
import paoding.utility.utils as utils
import paoding.utility.interval_arithmetic as ia
import paoding.utility.simulated_propagation as simprop
import math
import numpy as np
import random


# Saliency-only method
def pruning_baseline(model, big_map, prune_percentage=None,
                     neurons_manipulated=None,
                     saliency_matrix=None,
                     recursive_pruning=False,
                     bias_aware=False):
    # Load the parameters and configuration of the input model
    (w, g) = utils.load_param_and_config(model)

    num_layers = len(model.layers)
    total_pruned_count = 0
    layer_idx = 0

    pruned_pairs = []

    if neurons_manipulated is None:
        neurons_manipulated = []

    if saliency_matrix is None:
        saliency_matrix = []

    while layer_idx < num_layers - 1:
        pruned_pairs.append([])
        if len(neurons_manipulated) < layer_idx+1:
            neurons_manipulated.append([])
        saliency_matrix.append(None)
        # Exclude non FC layers
        if "dense" in model.layers[layer_idx].name:
            # print("Pruning Operation Looking at Layer", layer_idx)

            num_prev_neurons = len(w[layer_idx][0])
            num_curr_neurons = len(w[layer_idx][0][0])
            num_next_neurons = len(w[layer_idx + 1][0][0])

            # curr_weights_neuron_as_rows records the weights parameters originating from the prev layer
            curr_weights_neuron_as_rows = np.zeros((num_curr_neurons, num_prev_neurons))
            for idx_neuron in range(0, num_curr_neurons):
                for idx_prev_neuron in range(0, num_prev_neurons):
                    curr_weights_neuron_as_rows[idx_neuron][idx_prev_neuron] = w[layer_idx][0][idx_prev_neuron][
                        idx_neuron]

            # next_weights_neuron_as_rows records the weights parameters connecting to the next layer
            next_weights_neuron_as_rows = w[layer_idx + 1][0]

            if saliency_matrix[layer_idx] is None:

                print(" >> Building saliency matrix for layer "+str(layer_idx)+"...")
                if bias_aware:
                    # w[layer_idx][1] records the bias per each neuron in the current layer
                    saliency_matrix[layer_idx] = saliency.build_saliency_matrix_with_bias(curr_weights_neuron_as_rows,
                                                                           next_weights_neuron_as_rows,
                                                                           w[layer_idx][1])
                else:
                    saliency_matrix[layer_idx] = saliency.build_saliency_matrix(curr_weights_neuron_as_rows,
                                                                 next_weights_neuron_as_rows)
            else:
                print(" >> Skip building saliency matrix: saliency matrix for layer", layer_idx, "exists.")

            import pandas as pd
            df = pd.DataFrame(data=saliency_matrix[layer_idx])

            # find the candidates neuron to be pruned according to the saliency
            if prune_percentage is not None:
                num_candidates_to_gen = math.ceil(prune_percentage * num_curr_neurons)
            else:
                num_candidates_to_gen = 1

            top_candidates = utils.get_pairs_with_least_saliency(df, neurons_manipulated[layer_idx], num_candidates=num_candidates_to_gen)

            # Just return if there is no candidates to prune
            if len(top_candidates) == 0:
                return model, neurons_manipulated, [], saliency_matrix

            # Now let's process the top_candidate first:
            #   top_candidate is a list of Series, with key as multi-index, and value as saliency
            # and we are going to transform that list into a dictionary to facilitate further ajustment

            pruning_pairs_curr_layer_baseline = []

            for idx_candidate, pruning_candidate in enumerate(top_candidates):
                # Extract the indexes of pruning nodes as a tuple (corr. score is no longer useful since this step)
                (node_a, node_b) = pruning_candidate.index.values[0]
                ''' 
                # CHANGE on Commit db9c736, to make it consistent with paper
                # To standarise the pruning operation for the same pair: we always prune the node with smaller index off
                if node_a > node_b:
                    temp = node_a
                    node_a = node_b
                    node_b = temp
                '''
                pruning_pairs_curr_layer_baseline.append((node_a, node_b))

                # Change all weight connecting from node_a to the next layers as the sum of node_a and node_b's ones
                #    & Reset all weight connecting from node_b to ZEROs
                # RECALL: next_weights_neuron_as_rows = w[layer_idx+1][0] ([0] for weight and [1] for bias)
                for i in range(0, num_next_neurons):
                    w[layer_idx + 1][0][node_a][i] = w[layer_idx + 1][0][node_b][i] + w[layer_idx + 1][0][node_a][i]
                    w[layer_idx + 1][0][node_b][i] = 0
                total_pruned_count += 1

                # If recursive mode is enabled, the affected neuron (node_a) in the current epoch still 
                #   get a chance to be considered in the next epoch. The pruned one (node_b) won't be 
                #   considered any longer because all its parameters have been zeroed out.
                if recursive_pruning:
                    if neurons_manipulated[layer_idx] is not None:
                        neurons_manipulated[layer_idx].remove(node_a)

            # Save the modified parameters to the model
            model.layers[layer_idx + 1].set_weights(w[layer_idx + 1])

            pruned_pairs[layer_idx].extend(pruning_pairs_curr_layer_baseline)
        layer_idx += 1
    print("Pruning accomplished -", total_pruned_count, "units have been pruned")
    return model, neurons_manipulated, pruned_pairs, saliency_matrix


# Our greedy method without stochastic heuristic
def pruning_greedy(model, big_map, prune_percentage,
                   cumulative_impact_intervals,
                   pooling_multiplier=1,
                   neurons_manipulated=None,
                   hyperparamters=(0.5, 0.5),
                   recursive_pruning=True,
                   bias_aware=False,
                   kaggle_credit=False):
    # Load the parameters and configuration of the input model
    (w, g) = utils.load_param_and_config(model)

    num_layers = len(model.layers)
    total_pruned_count = 0
    layer_idx = 0

    pruned_pairs = []
    pruning_pairs_dict_overall_scores =[]


    if neurons_manipulated is None:
        neurons_manipulated = []

    e_ij_matrix = []

    while layer_idx < num_layers - 1:

        cumul_impact_ints_curr_layer = None

        pruned_pairs.append([])
        if len(neurons_manipulated) < layer_idx+1:
            neurons_manipulated.append([])
        e_ij_matrix.append(None)
        pruning_pairs_dict_overall_scores.append(None)

        # Exclude non FC layers
        if "dense" in model.layers[layer_idx].name:
            # print("Pruning Operation Looking at Layer", layer_idx)

            num_prev_neurons = len(w[layer_idx][0])
            num_curr_neurons = len(w[layer_idx][0][0])
            num_next_neurons = len(w[layer_idx + 1][0][0])

            # curr_weights_neuron_as_rows records the weights parameters originating from the prev layer
            curr_weights_neuron_as_rows = np.zeros((num_curr_neurons, num_prev_neurons))
            for idx_neuron in range(0, num_curr_neurons):
                for idx_prev_neuron in range(0, num_prev_neurons):
                    curr_weights_neuron_as_rows[idx_neuron][idx_prev_neuron] = w[layer_idx][0][idx_prev_neuron][
                        idx_neuron]

            # next_weights_neuron_as_rows records the weights parameters connecting to the next layer
            next_weights_neuron_as_rows = w[layer_idx + 1][0]

            print(" >> Building saliency matrix for layer " + str(layer_idx) + "...")
            if bias_aware:
                # w[layer_idx][1] records the bias per each neuron in the current layer
                e_ij_matrix[layer_idx] = saliency.build_saliency_matrix_with_bias(curr_weights_neuron_as_rows,
                                                                       next_weights_neuron_as_rows,
                                                                       w[layer_idx][1])
            else:
                e_ij_matrix[layer_idx] = saliency.build_saliency_matrix(curr_weights_neuron_as_rows,
                                                             next_weights_neuron_as_rows)

            import pandas as pd
            df = pd.DataFrame(data=e_ij_matrix[layer_idx])

            # find the candidates neuron to be pruned according to the saliency
            if prune_percentage is not None:
                num_candidates_to_gen = math.ceil(prune_percentage * num_curr_neurons)
            else:
                num_candidates_to_gen = 1
            # find the candidates neuron to be pruned according to the saliency
            top_candidates = utils.get_pairs_with_least_saliency(df, neurons_manipulated[layer_idx],
                                                                 num_candidates=num_candidates_to_gen* pooling_multiplier)

            # Just return if there is no candidates to prune
            if len(top_candidates) == 0:
                return model, neurons_manipulated, [], cumulative_impact_intervals, pruning_pairs_dict_overall_scores

            # Now let's process the top_candidate first:
            #   top_candidate is a list of Series, with key as multi-index, and value as saliency
            # and we are going to transform that list into a dictionary to facilitate further ajustment

            pruning_pairs_curr_layer_confirmed = []
            pruning_pairs_dict_curr_layer_l1_score = {}
            pruning_pairs_dict_curr_layer_entropy_score = {}

            pruning_pairs_dict_overall_scores[layer_idx] ={}

            for idx_candidate, pruning_candidate in enumerate(top_candidates):
                # Extract the indexes of pruning nodes as a tuple (corr. score is no longer useful since this step)
                (node_a, node_b) = pruning_candidate.index.values[0]
                # print(" >> Looking into", (node_a, node_b))
                '''
                # CHANGE on Commit db9c736, to make it consistent with paper
                # To standarise the pruning operation for the same pair: we always prune the node with smaller index off
                if node_a > node_b:
                    temp = node_a
                    node_a = node_b
                    node_b = temp
                '''
                # Below is the hill climbing algorithm to update the top_candidate by dividing the original saliency by
                #   the l1-norm of the budget preservation list (the higher the better)
                pruning_impact_as_interval_next_layer = simprop.calculate_impact_of_pruning_next_layer(model, big_map,
                                                                                            [(node_a, node_b)], layer_idx,
                                                                                            kaggle_credit=kaggle_credit)

                # Check is cumulative_impact_interval is none or not, not none means there is already some cumulative impact
                #   caused by previous pruning actions
                if cumul_impact_ints_curr_layer is not None:
                    pruning_impact_as_interval_next_layer = ia.interval_list_add(pruning_impact_as_interval_next_layer,
                                                                      cumul_impact_ints_curr_layer)

                pruning_impact_as_interval_output_layer = simprop.calculate_bounds_of_output(model,
                                                                                             pruning_impact_as_interval_next_layer,
                                                                                             layer_idx + 1)

                big_L = utils.l1_norm_of_intervals(pruning_impact_as_interval_output_layer)
                # Use sigmoid logistic to normalize
                big_L = 1 / (1 + math.exp(-1 * big_L))

                big_ENT = utils.interval_based_entropy(pruning_impact_as_interval_output_layer, similarity_criteria=0.9)
                # Use sigmoid logistic to normalize
                big_ENT = 1 / (1 + math.exp(-1 * big_ENT))
                # Now we are going re-sort the saliency according to the utilization situation of each pair pruning

                pruning_pairs_dict_curr_layer_l1_score[(node_a, node_b)] = big_L
                # Avoid entropy equals to zero
                pruning_pairs_dict_curr_layer_entropy_score[(node_a, node_b)] = big_ENT

                (alpha, beta) = hyperparamters
                print((node_a, node_b), "Ent:", big_ENT)
                # pruning_pairs_dict_overall_scores[(node_a, node_b)] = pruning_candidate.values[0] * (big_L * alpha + big_ENT * beta)
                pruning_pairs_dict_overall_scores[layer_idx][(node_a, node_b)] = big_L * alpha + big_ENT * beta

            count = 0

            pruning_pairs_dict_overall_scores[layer_idx] = dict(sorted(pruning_pairs_dict_overall_scores[layer_idx].items(), key=lambda item: item[1]))
            for pair in pruning_pairs_dict_overall_scores[layer_idx]:
                if count < num_candidates_to_gen:
                    pruning_pairs_curr_layer_confirmed.append(pair)

                    # If recursive mode is enabled, the affected neuron (node_a) in the current epoch still 
                    #   get a chance to be considered in the next epoch. The pruned one (node_b) won't be 
                    #   considered any longer because all its parameters have been zeroed out.
                    if recursive_pruning:
                        (neuron_a, neuron_b) = pair
                        neurons_manipulated[layer_idx].remove(neuron_a)
                else:
                    # Drop that pair from the neurons_manipulated list and enable re-considering in future epoch
                    (neuron_a, neuron_b) = pair
                    neurons_manipulated[layer_idx].remove(neuron_a)
                    neurons_manipulated[layer_idx].remove(neuron_b)
                count += 1

            # Here we evaluate the impact to the output layer
            if cumul_impact_ints_curr_layer is None:
                cumul_impact_ints_curr_layer = simprop.calculate_impact_of_pruning_next_layer(model, big_map, pruning_pairs_curr_layer_confirmed, layer_idx)
            else:
                cumul_impact_ints_curr_layer = ia.interval_list_add(cumul_impact_ints_curr_layer,
                                                                              simprop.calculate_impact_of_pruning_next_layer(model, big_map,
                                                                                                                             pruning_pairs_curr_layer_confirmed,
                                                                                                                             layer_idx,
                                                                                                                             kaggle_credit=kaggle_credit))

            if cumulative_impact_intervals is None:
                cumulative_impact_intervals = simprop.calculate_bounds_of_output(model, cumul_impact_ints_curr_layer, layer_idx+1)
            else:
                cumulative_impact_intervals= ia.interval_list_add(cumulative_impact_intervals,
                                                                  simprop.calculate_bounds_of_output(model,
                                                                                                     cumul_impact_ints_curr_layer,
                                                                                                     layer_idx+1))

            print(" >> DEBUG: len(cumulative_impact_curr_layer_pruning_to_next_layer):", len(cumul_impact_ints_curr_layer))
            print(" >> DEBUG: len(cumulative_impact_to_output_layer):", len(cumulative_impact_intervals))

            # Now let's do pruning (simulated, by zeroing out weights but keeping neurons in the network)
            for (node_a, node_b) in pruning_pairs_curr_layer_confirmed:
                # Change all weight connecting from node_b to the next layers as the sum of node_a and node_b's ones
                #    & Reset all weight connecting from node_a to ZEROs
                # RECALL: next_weights_neuron_as_rows = w[layer_idx+1][0] ([0] for weight and [1] for bias)
                for i in range(0, num_next_neurons):
                    w[layer_idx + 1][0][node_a][i] = w[layer_idx + 1][0][node_b][i] + w[layer_idx + 1][0][node_a][i]
                    w[layer_idx + 1][0][node_b][i] = 0
                total_pruned_count += 1
            # Save the modified parameters to the model
            model.layers[layer_idx + 1].set_weights(w[layer_idx + 1])

            pruned_pairs[layer_idx].extend(pruning_pairs_curr_layer_confirmed)

            # TEMP IMPLEMENTATION STARTS HERE
            if not kaggle_credit:
                big_map = simprop.get_definition_map(model, definition_dict=big_map, input_interval=(0, 1))
            else:
                big_map = simprop.get_definition_map(model, definition_dict=big_map, input_interval=(-5, 5))

            print("Pruning layer #", layer_idx, "completed, updating definition hash map...")
            # TEMP IMPLEMENTATION ENDS HERE

        layer_idx += 1

    print(" >> DEBUG: size of cumulative impact total", len(cumulative_impact_intervals))
    print("Pruning accomplished -", total_pruned_count, "units have been pruned")
    return model, neurons_manipulated, pruned_pairs, cumulative_impact_intervals, pruning_pairs_dict_overall_scores


def pruning_stochastic(model, big_map, prune_percentage,
                      cumulative_impact_intervals,
                      neurons_manipulated=None,
                      target_scores=None,
                      hyperparamters=(0.5, 0.5),
                      recursive_pruning=True,
                      bias_aware=False,
                      kaggle_credit=False):
    # Load the parameters and configuration of the input model
    (w, g) = utils.load_param_and_config(model)

    num_layers = len(model.layers)
    total_pruned_count = 0
    layer_idx = 0

    pruned_pairs = []
    pruning_pairs_dict_overall_scores =[]


    if neurons_manipulated is None:
        neurons_manipulated = []

    if target_scores is None:
        target_scores = []

    e_ij_matrix = []

    while layer_idx < num_layers - 1:

        cumul_impact_ints_curr_layer = None

        pruned_pairs.append([])

        if len(neurons_manipulated) < layer_idx + 1:
            neurons_manipulated.append([])
        if len(target_scores) < layer_idx + 1:
            target_scores.append(-1)

        e_ij_matrix.append(None)
        pruning_pairs_dict_overall_scores.append(None)

        # Exclude non FC layers
        if "dense" in model.layers[layer_idx].name:
            # print("Pruning Operation Looking at Layer", layer_idx)

            num_prev_neurons = len(w[layer_idx][0])
            num_curr_neurons = len(w[layer_idx][0][0])
            num_next_neurons = len(w[layer_idx + 1][0][0])

            # curr_weights_neuron_as_rows records the weights parameters originating from the prev layer
            curr_weights_neuron_as_rows = np.zeros((num_curr_neurons, num_prev_neurons))
            for idx_neuron in range(0, num_curr_neurons):
                for idx_prev_neuron in range(0, num_prev_neurons):
                    curr_weights_neuron_as_rows[idx_neuron][idx_prev_neuron] = w[layer_idx][0][idx_prev_neuron][
                        idx_neuron]

            # next_weights_neuron_as_rows records the weights parameters connecting to the next layer
            next_weights_neuron_as_rows = w[layer_idx + 1][0]

            print(" >> Building saliency matrix for layer " + str(layer_idx) + "...")
            if bias_aware:
                # w[layer_idx][1] records the bias per each neuron in the current layer
                e_ij_matrix[layer_idx] = saliency.build_saliency_matrix_with_bias(curr_weights_neuron_as_rows,
                                                                       next_weights_neuron_as_rows,
                                                                       w[layer_idx][1])
            else:
                e_ij_matrix[layer_idx] = saliency.build_saliency_matrix(curr_weights_neuron_as_rows,
                                                             next_weights_neuron_as_rows)

            import pandas as pd
            df = pd.DataFrame(data=e_ij_matrix[layer_idx])

            # find the candidates neuron to be pruned according to the saliency
            if prune_percentage is not None:
                num_candidates_to_gen = math.ceil(prune_percentage * num_curr_neurons)
            else:
                num_candidates_to_gen = 1
            # find the candidates neuron to be pruned according to the saliency
            top_candidates = utils.get_all_pairs_by_saliency(df, neurons_manipulated[layer_idx])

            # Just return if there is no candidates to prune
            if len(top_candidates) == 0:
                return model, neurons_manipulated, target_scores, [], cumulative_impact_intervals, pruning_pairs_dict_overall_scores

            # Now let's process the top_candidate first:
            #   top_candidate is a list of Series, with key as multi-index, and value as saliency
            # and we are going to transform that list into a dictionary to facilitate further ajustment

            pruning_pairs_curr_layer_confirmed = []
            count = 0

            pruning_pairs_dict_overall_scores[layer_idx] ={}

            # A workaround if pruned candidates is less than num_candidate_to_prune after a walking
            #  then we need to re-walk again until the number of units to be pruned reaches target
            while (count < num_candidates_to_gen):
                for idx_candidate, pruning_candidate in enumerate(top_candidates):
                    # Extract the indexes of pruning nodes as a tuple (corr. score is no longer useful since this step)
                    (node_a, node_b) = pruning_candidate.index.values[0]
                    # print(" >> Looking into", (node_a, node_b))

                    '''
                    # CHANGE on Commit db9c736, to make it consistent with paper
                    # To standarise the pruning operation for the same pair: we always prune the node with smaller index off
                    if node_a > node_b:
                        temp = node_a
                        node_a = node_b
                        node_b = temp
                    '''

                    if count < num_candidates_to_gen:

                        # Below is the hill climbing algorithm to update the top_candidate by dividing the original saliency by
                        #   the l1-norm of the budget preservation list (the higher the better)
                        pruning_impact_as_interval_next_layer = simprop.calculate_impact_of_pruning_next_layer(model, big_map,
                                                                                                    [(node_a, node_b)], layer_idx,
                                                                                                    kaggle_credit=kaggle_credit)

                        # Check is cumulative_impact_interval is none or not, not none means there is already some cumulative impact
                        #   caused by previous pruning actions
                        if cumul_impact_ints_curr_layer is not None:
                            pruning_impact_as_interval_next_layer = ia.interval_list_add(pruning_impact_as_interval_next_layer,
                                                                              cumul_impact_ints_curr_layer)

                        pruning_impact_as_interval_output_layer = simprop.calculate_bounds_of_output(model,
                                                                                                     pruning_impact_as_interval_next_layer,
                                                                                                     layer_idx + 1)

                        big_L = utils.l1_norm_of_intervals(pruning_impact_as_interval_output_layer)
                        # Use sigmoid logistic to normalize
                        big_L = 1 / (1 + math.exp(-1 * big_L))

                        big_ENT = utils.interval_based_entropy(pruning_impact_as_interval_output_layer, similarity_criteria=0.9)
                        # Use sigmoid logistic to normalize
                        big_ENT = 1 / (1 + math.exp(-1 * big_ENT))
                        # Now we are going re-sort the saliency according to the utilization situation of each pair pruning

                        (alpha, beta) = hyperparamters
                        # print((node_a, node_b), "Ent:", big_ENT)

                        curr_score = big_L * alpha + big_ENT * beta

                        # Accept the first sample by-default, or a sample with better (smaller) score
                        if target_scores[layer_idx] == -1 or curr_score <= target_scores[layer_idx]:

                            target_scores[layer_idx] = curr_score
                            pruning_pairs_curr_layer_confirmed.append((node_a, node_b))

                            # If recursive mode is enabled, the affected neuron (node_a) in the current epoch still 
                            #   get a chance to be considered in the next epoch. The pruned one (node_b) won't be 
                            #   considered any longer because all its parameters have been zeroed out.
                            if recursive_pruning:
                                if node_a in neurons_manipulated[layer_idx]:
                                    neurons_manipulated[layer_idx].remove(node_a)

                            count += 1
                            pruning_pairs_dict_overall_scores[layer_idx][(node_a, node_b)] = target_scores[layer_idx]

                            print(" [DEBUG]", bcolors.OKGREEN, "Accepting", bcolors.ENDC, (node_a, node_b), curr_score)

                        # Then we use simulated annealing algorithm to determine if we accept the next pair in the pruning list
                        else:
                            # Progress is a variable that grows from 0 to 1
                            progress = len(neurons_manipulated[layer_idx])/num_curr_neurons

                            # Define a temperature decending linearly with progress goes on (add 0.0001 to avoid divide-by-zero issue)
                            temperature = 1.0001 - progress

                            # Calculate the delta of score (should be a positive value because objective is minimum)
                            delta_score = curr_score - target_scores[layer_idx]
                            prob_sim_annealing = math.exp(-1 * delta_score / temperature)
                            prob_random = random.random()

                            # Higher probability of simulated annealing, easilier to accept a bad choice
                            if prob_random < prob_sim_annealing:

                                target_scores[layer_idx] = curr_score
                                pruning_pairs_curr_layer_confirmed.append((node_a, node_b))

                                # If recursive mode is enabled, the affected neuron (node_a) in the current epoch still 
                                #   get a chance to be considered in the next epoch. The pruned one (node_b) won't be 
                                #   considered any longer because all its parameters have been zeroed out.
                                if recursive_pruning:
                                    if node_a in neurons_manipulated[layer_idx]:
                                        neurons_manipulated[layer_idx].remove(node_a)

                                count += 1
                                pruning_pairs_dict_overall_scores[layer_idx][(node_a, node_b)] = curr_score

                                print(" [DEBUG]", bcolors.OKGREEN, "Accepting (stochastic)", bcolors.ENDC, (node_a, node_b), "despite the score", 
                                    round(curr_score, 6), "because the probability", round(prob_random, 6), "<=", round(prob_sim_annealing, 6))

                            else:
                                print(" [DEBUG]", bcolors.FAIL, "Reject", bcolors.ENDC, (node_a, node_b), "because the score", 
                                    round(curr_score, 6), ">", round(target_scores[layer_idx], 6), "and random prob. doesn't satisfy", round(prob_sim_annealing, 6))
                                # Drop that pair from the neurons_manipulated list and enable re-considering in future epoch
                                if node_b in neurons_manipulated[layer_idx]:
                                    neurons_manipulated[layer_idx].remove(node_b)
                                if node_a in neurons_manipulated[layer_idx]:
                                    neurons_manipulated[layer_idx].remove(node_a)
                    else:
                        # Drop that pair from the neurons_manipulated list and enable re-considering in future epoch
                        if node_b in neurons_manipulated[layer_idx]:
                            neurons_manipulated[layer_idx].remove(node_b)
                        if node_a in neurons_manipulated[layer_idx]:
                            neurons_manipulated[layer_idx].remove(node_a)

                if (count < num_candidates_to_gen):
                    print(" >> Insufficient number of pruning candidates, walk again ...")

            # Here we evaluate the impact to the output layer
            if cumul_impact_ints_curr_layer is None:
                cumul_impact_ints_curr_layer = simprop.calculate_impact_of_pruning_next_layer(model, big_map, pruning_pairs_curr_layer_confirmed, layer_idx)
            else:
                cumul_impact_ints_curr_layer = ia.interval_list_add(cumul_impact_ints_curr_layer,
                                                                              simprop.calculate_impact_of_pruning_next_layer(model, big_map,
                                                                                                                             pruning_pairs_curr_layer_confirmed,
                                                                                                                             layer_idx,
                                                                                                                             kaggle_credit=kaggle_credit))

            if cumulative_impact_intervals is None:
                cumulative_impact_intervals = simprop.calculate_bounds_of_output(model, cumul_impact_ints_curr_layer, layer_idx+1)
            else:
                cumulative_impact_intervals= ia.interval_list_add(cumulative_impact_intervals,
                                                                  simprop.calculate_bounds_of_output(model,
                                                                                                     cumul_impact_ints_curr_layer,
                                                                                                     layer_idx+1))

            # print(" >> DEBUG: len(cumulative_impact_curr_layer_pruning_to_next_layer):", len(cumul_impact_ints_curr_layer))
            # print(" >> DEBUG: len(cumulative_impact_to_output_layer):", len(cumulative_impact_intervals))

            # Now let's do pruning (simulated, by zeroing out weights but keeping neurons in the network)
            for (node_a, node_b) in pruning_pairs_curr_layer_confirmed:
                # Change all weight connecting from node_b to the next layers as the sum of node_a and node_b's ones
                #    & Reset all weight connecting from node_a to ZEROs
                # RECALL: next_weights_neuron_as_rows = w[layer_idx+1][0] ([0] for weight and [1] for bias)
                for i in range(0, num_next_neurons):
                    w[layer_idx + 1][0][node_a][i] = w[layer_idx + 1][0][node_b][i] + w[layer_idx + 1][0][node_a][i]
                    w[layer_idx + 1][0][node_b][i] = 0
                total_pruned_count += 1
            # Save the modified parameters to the model
            model.layers[layer_idx + 1].set_weights(w[layer_idx + 1])

            pruned_pairs[layer_idx].extend(pruning_pairs_curr_layer_confirmed)

            # TEMP IMPLEMENTATION STARTS HERE
            if not kaggle_credit:
                big_map = simprop.get_definition_map(model, definition_dict=big_map, input_interval=(0, 1))
            else:
                big_map = simprop.get_definition_map(model, definition_dict=big_map, input_interval=(-5, 5))

            print("Pruning layer #", layer_idx, "completed, updating definition hash map...")
            # TEMP IMPLEMENTATION ENDS HERE

        layer_idx += 1

    print(" >> DEBUG: size of cumulative impact total", len(cumulative_impact_intervals))
    print("Pruning accomplished -", total_pruned_count, "units have been pruned")
    return model, neurons_manipulated, target_scores, pruned_pairs, cumulative_impact_intervals, pruning_pairs_dict_overall_scores
