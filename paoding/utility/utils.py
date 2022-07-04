#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

import math, os, errno
from operator import add
import numpy as np
import progressbar


def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        # for j in range(0, i + 1):
        #    pairs_to_drop.add((cols[i], cols[j]))
        pairs_to_drop.add((cols[i], cols[i]))
    return pairs_to_drop

def create_dir_if_not_exist(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except:
            if exec.errno != errno.EEXIST:
                raise
    return None


def load_param_and_config(model, debugging_output=False):
    # Take a 3 layer MLP as example:
    # "w" is a list, where "w[0]" is empty as it specifies the input layer
    #  "w[1]" contains paramters for the 1st hidden layer, where
    #   "w[1][0] is in shape of 784x128" and w[1][1] is in shape of 128x1 (bias);
    #  "w[2]" contains paramters for the 2nd hidden layer, where
    #   "w[2][0] is in shape of 128x10" and w[2][1] is in shape of 10x1 (bias);

    g = []
    w = []
    layer_index = 0
    for layer in model.layers:
        g.append(layer.get_config())
        w.append(layer.get_weights())
        if "dense" in layer.name:
            num_units = g[layer_index]['units']
            num_prev_neurons = len(w[layer_index][0])
            if debugging_output:
                print("TYPE OF ACTIVATION: ", g[layer_index]['activation'])
                print("CURRENT LAYER: ", g[layer_index]['name'])
                print("NUM OF UNITS: ", num_units)
                print("NUM OF CONNECTION PER UNIT: ", num_prev_neurons)
        layer_index += 1
    return (w, g)

def get_pairs_with_least_saliency(df, neurons_manipulated, num_candidates=5):
    au_corr = df.unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop)

    # Smaller correlation value means two units are more similar with each other
    au_corr = au_corr.sort_values(ascending=True)

    i = 0

    # Add the pair with the least saliency into the list to prune
    list_to_prune = []

    # neurons_manipulated records all the units has been involved in pruning, to avoid double pruning
    if neurons_manipulated is None:
        neurons_manipulated = []

    # Return the requested number of pruning pairs or all possible pairs, which is smallest
    while len(list_to_prune) < num_candidates and i < len(au_corr) - 1:
        i += 1
        curr_index_pair = list(au_corr[i:i + 1].index.tolist()[0])
        if curr_index_pair[0] in neurons_manipulated or curr_index_pair[1] in neurons_manipulated:
            continue
        for neuron in curr_index_pair:
            neurons_manipulated.append(neuron)
        list_to_prune.append(au_corr[i:i + 1])
    return list_to_prune

def get_all_pairs_by_saliency(df, neurons_manipulated):
    au_corr = df.unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop)

    # Smaller correlation value means two units are more similar with each other
    au_corr = au_corr.sort_values(ascending=True)

    i = 0

    # Add the pair with the least saliency into the list to prune
    list_to_prune = []

    # neurons_manipulated records all the units has been involved in pruning, to avoid double pruning
    if neurons_manipulated is None:
        neurons_manipulated = []

    # Return the requested number of pruning pairs or all possible pairs, which is smallest
    while i < len(au_corr) - 1:
        i += 1
        curr_index_pair = list(au_corr[i:i + 1].index.tolist()[0])
        if curr_index_pair[0] in neurons_manipulated or curr_index_pair[1] in neurons_manipulated:
            continue
        for neuron in curr_index_pair:
            neurons_manipulated.append(neuron)
        list_to_prune.append(au_corr[i:i + 1])
    return list_to_prune

def zero_out_k_smallest_elements(matrix, k):
    matrix_flatten = matrix.reshape(-1)

    if len(matrix_flatten) <= k:
        print("The value of k=", k, " exceeds the size of weight matrix, k has been resize to", len(matrix_flatten))
        k = len(matrix_flatten)

    indices = np.argsort(matrix_flatten)[:k]
    for i in indices:
        matrix_flatten[i] = 0
    print(" >> DEBUG: num of zero elements in weight matrix is", np.count_nonzero(matrix_flatten==0))
    return matrix_flatten.reshape(matrix.shape)


def get_pairs_with_least_saliency_robust_preserved(df, robustness_impact, next_parameters, k=10, num_candidates=5):
    au_corr = df.unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop)

    # Smaller correlation value means two units are more similar with each other
    au_corr = au_corr.sort_values(ascending=True)

    index = 0

    # Add the pair with the least saliency into the list to prune
    list_to_prune = [au_corr[0:1]]
    # neurons_manipulated records all the units has been involved in pruning, to avoid double pruning
    neurons_manipulated = list(au_corr[0:1].index.tolist()[0])

    # Remove the pair that has been added into the list to prune
    au_corr = au_corr.drop(au_corr.index[0:1])

    # cumulative_impact records the bounds of cumulative impact (delta) in coefficient format
    cumulative_impact = [0.0 for i in range(0, len(robustness_impact[0][0]))]

    #print(" >> target at pruning", num_candidates, "units at the current layer ")

    bar = progressbar.ProgressBar(maxval=num_candidates,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    # Return the requested number of pruning pairs or all possible pairs, which is smallest
    while len(list_to_prune) < num_candidates and len(au_corr) > 0:

        # print(" >>> looking for k pairs for the next pruning candidate")
        # Collect k pairs valid for the current prune list
        idx_for_k = 0
        k_candidate = []
        while idx_for_k < k and len(au_corr)>idx_for_k:
            curr_pair = list(au_corr[idx_for_k:idx_for_k + 1].index.tolist()[0])
            if curr_pair[0] in neurons_manipulated or curr_pair[1] in neurons_manipulated:
                # The next_pair is not valid anymore as one of its units has been involved in pruning
                #   therefore we remove it from the candidate list
                au_corr = au_corr.drop(au_corr.index[idx_for_k:idx_for_k + 1])
            else:
                k_candidate.append(au_corr[idx_for_k:idx_for_k + 1])
                idx_for_k += 1

        index_of_min_impact = -1
        min_impact = -1
        best_option_cumulative_impact = []

        # for debugging purpose
        list_of_impact = []

        # print(" >>> found k pairs for the next pruning candidate")

        for index, item in enumerate(k_candidate):
            curr_pair_indexes = list(item.index)[0]
            robustness_impact_by_curr_pairs = robustness_impact[curr_pair_indexes[0]][curr_pair_indexes[1]]
            rob_impact_curr_upper_bound = []
            parameters_from_unit_a_to_next_layer = next_parameters[curr_pair_indexes[0]]
            for i, j in enumerate(robustness_impact_by_curr_pairs):
                sign = 1.0
                if parameters_from_unit_a_to_next_layer[i] < 0:
                    sign = -1.0j
                rob_impact_curr_upper_bound.append(j[1] * sign)
            candidate_cumulative_impact = list(map(add, rob_impact_curr_upper_bound, cumulative_impact))
            impact_as_l2_norm = np.linalg.norm(candidate_cumulative_impact)
            list_of_impact.append(impact_as_l2_norm)
            if min_impact < 0 or (min_impact > 0 and min_impact > impact_as_l2_norm):
                min_impact = impact_as_l2_norm
                index_of_min_impact = index
                best_option_cumulative_impact = candidate_cumulative_impact

        cumulative_impact = best_option_cumulative_impact
        # print("k_cand:", len(k_candidate), "- au_corr", len(au_corr), "- index_min_impact", index_of_min_impact, " - list_of_impact", list_of_impact)
        neurons_manipulated += list(au_corr[index_of_min_impact:index_of_min_impact + 1].index.tolist()[0])
        list_to_prune.append(au_corr[index_of_min_impact:index_of_min_impact + 1])
        au_corr = au_corr.drop(au_corr.index[index_of_min_impact:index_of_min_impact+1])

        #print(".", end="")
        #print(" >> a new candidate of pruning found, ", int(num_candidates-len(list_to_prune)), "left...")
        bar.update(len(list_to_prune))
    bar.finish()
    print("")
    return list_to_prune, cumulative_impact


def l1_norm_of_intervals(interval_list):
    list_lo = []
    list_hi = []
    for interval in interval_list:
        list_lo.append(interval[0])
        list_hi.append(interval[1])
    arr_hi = np.array(list_hi)
    arr_lo = np.array(list_lo)
    return np.linalg.norm(arr_hi - arr_lo, ord=1)


def union_of_interval(interval_list):
    union_lo = 0
    union_hi = 0
    for interval in interval_list:
        int_lo, int_hi = interval
        if union_lo > int_lo:
            union_lo = int_lo
        if union_hi < int_hi:
            union_hi = int_hi
    return (union_lo, union_hi)


def calculate_similarity_of_two_intervals(interval_a, interval_b, union_interval):
    a_lo, a_hi = interval_a
    b_lo, b_hi = interval_b
    union_lo, union_hi = union_interval
    if union_hi == union_lo:
        union_hi += 0.001
    sim_ab = 1 - 0.5 * (abs(a_lo - b_lo) + abs(a_hi - b_hi)) / (union_hi - union_lo)
    return sim_ab


# Calculating entropy of a group of intervals. The entropy tends to be close to 0 if all input
#  intervals are similar according to the similarity_criteria provided. Otherwise the entropy
#  keeps growing along with intervals grow differently.

def interval_based_entropy(interval_list, similarity_criteria=0.8):
    # Restrict similarity criteria to be in interval [0, 1]
    if similarity_criteria < 0:
        similarity_criteria = 0
    if similarity_criteria > 1:
        similarity_criteria = 1

    size_of_interval_list = len(interval_list)
    sim_matrix = generate_similarity_matrix(interval_list)
    # According to Dai et al. 2016, the entropy of a list of intervals is calculated as:
    #   ENT(B) = - big_sum(i) [ p(S_similarity_criteria_B(u_i)) * log p(S_similarity_criteria_B(u_i)) ]
    entropy = 0
    for index_i, u_i in enumerate(interval_list):
        p_s_b = 0
        for index_j, u_j in enumerate(interval_list):
            similarity = sim_matrix[index_i][index_j]
            if similarity >= similarity_criteria:
                p_s_b += 1
        p_s_b /= size_of_interval_list
        entropy -= p_s_b * math.log(p_s_b)
    return entropy


def generate_similarity_matrix(interval_list):
    size_of_interval_list = len(interval_list)
    union_interval = union_of_interval(interval_list)
    # Initialize with ones as similarity of 2 same intervals is 1
    similarity_matrix = np.ones((size_of_interval_list, size_of_interval_list))
    for idx_a in range(0, size_of_interval_list):
        for idx_b in range(0, idx_a):
            similarity_matrix[idx_a][idx_b] = calculate_similarity_of_two_intervals(interval_list[idx_a],
                                                                                    interval_list[idx_b],
                                                                                    union_interval)
            similarity_matrix[idx_b][idx_a] = similarity_matrix[idx_a][idx_b]
    return similarity_matrix



def main():
    interval_list_extreme = [(-50,-40), (-4,-3), (30,40), (140,250)]
    interval_list = [(-5,-4), (-4,-3), (3,4), (4,5)]
    interval_list_uniform = [(-5,4), (-4,5), (-4,4), (-4,5)]
    print(union_of_interval(interval_list))
    print(generate_similarity_matrix(interval_list))
    print(interval_based_entropy(interval_list, similarity_criteria=0.8))
    print(generate_similarity_matrix(interval_list_extreme))
    print(interval_based_entropy(interval_list_extreme, similarity_criteria=0.8))
    print(generate_similarity_matrix(interval_list_uniform))
    print(interval_based_entropy(interval_list_uniform, similarity_criteria=0.8))


if __name__ == "__main__":
    main()