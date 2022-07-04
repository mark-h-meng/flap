#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

import math


def interval_minus(a, b):
    if not len(a) == 2 or not len(b) == 2:
        raise Exception("Error: one of the inputs is not an interval - len(a):", len(a), "& len(b):", len(b))
    (a_lo, a_hi) = a
    (b_lo, b_hi) = b
    # Calculating (a - b)
    res_hi = a_hi - b_lo
    res_lo = a_lo - b_hi
    return (res_lo, res_hi)


def interval_add(a, b):
    if not len(a) == 2 or not len(b) == 2:
        raise Exception("Error: one of the inputs is not an interval - len(a):",
                        len(a), "& len(b):", len(b))
    (a_lo, a_hi) = a
    (b_lo, b_hi) = b
    # Calculating (a - b)
    res_hi = a_hi + b_hi
    res_lo = a_lo + b_lo
    return (res_lo, res_hi)


def interval_scale(a, k):
    if not len(a) == 2:
        raise Exception("Error: the inputs is not an interval - len(a):", len(a))
    (a_lo, a_hi) = a
    if k==0:
        return (0.0, 0.0)
    if k>0:
        res_hi = a_hi * k
        res_lo = a_lo * k
    else:
        res_lo = a_hi * k
        res_hi = a_lo * k
    return (res_lo, res_hi)


def interval_sum(interval_grp):
    res_hi = 0.0
    res_lo = 0.0
    for interval in interval_grp:
        if not len(interval) == 2:
            raise Exception("Error: one of the inputs in the list is not an interval - len(interval):",
                            len(interval))
        (a_lo, a_hi) = interval
        res_hi += a_hi
        res_lo += a_lo
    return (res_lo, res_hi)

def interval_list_add(list_a, list_b):
    if list_a is None:
        return list_b

    if list_b is None:
        return list_a

    if len(list_a) != len(list_b):
        raise Exception("Error: inconsistent length of two input lists - len(list_a):",
                        len(list_a), "& len(list_b):", len(list_b))
    res = []
    for index, item_a in enumerate(list_a):
        item_b = list_b[index]
        res.append(interval_add(item_a, item_b))
    return res


def forward_propogation(curr_layer, weight_params, bias_params, activation=True, relu_activation=True):
    # Obtain the size of the curr layer
    curr_size = len(curr_layer)
    # Obtain the size of the next layer
    res_size = len(bias_params)
    # Check the size of weight_params, should be (res_size * curr_size)
    if len(weight_params) != curr_size or len(weight_params[0]) != res_size:
        raise Exception("Error: wrong size of provided weight parameters",
                        str(len(weight_params))+"*"+str(len(weight_params[0])),
                        "expected, but curr_layer is in size of", len(weight_params),
                        "and bias_params is in size of", len(bias_params))

    result = []
    for i in range(0, res_size):
        output_hi = 0
        output_lo = 0
        for j in range(0, curr_size):
            (j_lo, j_hi) = interval_scale(curr_layer[j], weight_params[j][i])
            output_hi += j_hi
            output_lo += j_lo
        output_hi += bias_params[i]
        output_lo += bias_params[i]

        if activation and relu_activation:
            if output_hi < 0:
                output_hi = 0
            if output_lo < 0:
                output_lo = 0

        # simulate sigmoid activation below
        if activation and not relu_activation:
            output_hi =  1 / (1 + math.exp(-1 * output_hi))
            output_lo =  1 / (1 + math.exp(-1 * output_lo))

        # otherwise no activation needed (usually applicable to the final output layer)
        result.append((output_lo, output_hi))
    return result


def is_subset_interval(a, b):
    if not len(a) == 2 or not len(b) == 2:
        raise Exception("Error: one of the inputs is not an interval - len(a):",
                        len(a), "& len(b):", len(b))
    (a_lo, a_hi) = a
    (b_lo, b_hi) = b
    if a_lo > b_lo and a_hi < b_hi:
        return True
    else:
        return False


def check_budget_preservation(utilized_intervals, budget_intervals):
    if len(utilized_intervals) != len(budget_intervals) :
        raise Exception("Error: inconsistent size of intervals - len(utilized_intervals):",
                        len(utilized_intervals), "& len(budget_intervals):", len(budget_intervals))

    is_within_budget = [0 for i in budget_intervals]
    for index, interval in enumerate(utilized_intervals):
        if is_subset_interval(interval, budget_intervals[index]):
            is_within_budget[index] = 1
    return is_within_budget


# For test purpose
def main():
    (a, b) = (-0.9, 0.9)
    (c, d) = (-0.1, 0.1)
    (e, f) = (0.1, 0.9)

    b_params = [-1, 1]
    w_params = [[0.5, -0.5],
                [0.5, -0.5],
                [0.5, -0.5]]

    assert (-1.0, 1.0) == interval_sum([(a, b), (c, d)]), "Error occurred in interval sum operation"
    assert (-0.8, 1.8) == interval_add((a, b), (e, f)), "Error occurred in interval minus operation"
    assert (-1.0, 1.0) == interval_minus((a, b), (c, d)), "Error occurred in interval minus operation"
    assert (-1.0, 1.0) == interval_scale((c, d), 10), "Error occurred in interval scale operation"

    res_fp = forward_propogation([(a,b), (c,d), (e,f)], w_params, b_params, relu_activation=False)
    res_fp_testing = []
    for i in res_fp:
        (l, h) = i
        l = round(l,2)
        h = round(h,2)
        res_fp_testing.append((l,h))

    assert [(-1.45, -0.05), (0.05, 1.45)] == res_fp_testing, "Error occurred in forward propagation"

    budget = [(-0.5, 0.5),
              (-0.5, 0.5),
              (-0.5, 0.5)]

    utilized_budget = [(-0.4, 0.4),
                       (-0.5, 0.5),
                       (0, 0.4)]

    assert check_budget_preservation(utilized_budget, budget) == [1, 0, 1], \
        "Error occurred in checking budget preservation"
    assert interval_list_add(utilized_budget, budget) == [(-0.9, 0.9), (-1.0, 1.0), (-0.5, 0.9)] , \
        "Error occurred in adding two interval lists"

if __name__ == "__main__":
    main()