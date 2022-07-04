#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

import numpy as np


def remove_from_array(arr, index):
    list = arr.tolist()
    list.pop(index)
    return np.asarray(list)


def remove_row_from_2d_array(arr, row_index):
    list = arr.tolist()
    list.pop(row_index)
    return np.asarray(list)


def remove_column_from_2d_array(arr, column_index):
    list = arr.tolist()
    for row in list:
        row.pop(column_index)
    return np.asarray(list)


# For testing purpose
def main():
    sample_array = np.array([[0,0,1,2,3,4,5,6],
                          [1,0,1,2,3,4,5,6],
                          [2,0,1,2,3,4,5,6]])

    sample_array_rm_row = np.array([[0,0,1,2,3,4,5,6],
                          [2,0,1,2,3,4,5,6]])

    sample_array_rm_column = np.array([[0,0,1,2,4,5,6],
                          [1,0,1,2,4,5,6],
                          [2,0,1,2,4,5,6]])

    list0 = np.array([0,1,2,3,4,6])
    updated_list = remove_from_array(remove_from_array(sample_array[0], 6), 1)
    assert np.array_equal(updated_list, list0), "Error occurred in array deletion"

    list1 = remove_row_from_2d_array(sample_array, 1)
    assert np.array_equal(list1, sample_array_rm_row), "Error occurred in row deletion"

    list2 = remove_column_from_2d_array(sample_array, 4)
    assert np.array_equal(list2, sample_array_rm_column), "Error occurred in column deletion"

if __name__ == "__main__":
    main()