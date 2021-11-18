import numpy as np
import random
import itertools
import sys,os
import copy

#np.random.seed(42)
import numpy as np

TASK_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def print_dict(input_dict):
    for k,v in input_dict.items():
        print(k, v)

"""
    merge a list of lists
"""

def flatten_list(input_list):

    flat_list = list(itertools.chain.from_iterable(input_list))
    return flat_list



"""
sort a dictionary by value
"""

def sort_dict_value(sample_dict = None, reverse_mode = False, sort_index = 1):

    sorted_dict = sorted(sample_dict.items(), key=lambda x: x[sort_index], reverse = reverse_mode)

    return sorted_dict

def reverse_keys_values_dict(input_dict = None):

    output_dict = {}

    for k,v in input_dict.items():
        output_dict[v] = k

    return output_dict

def remove_and_create_dir(path):
    """ System call to rm -rf and then re-create a dir """

    dir = os.path.dirname(path)
    print('attempting to delete ', dir, ' path ', path)
    if os.path.exists(path):
        os.system("rm -rf " + path)
    os.system("mkdir -p " + path)
