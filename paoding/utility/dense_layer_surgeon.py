#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2022, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"


import enum
import tensorflow as tf
import numpy as np
import os, shutil

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


def trim_weights(model, pruned_pairs):
    (w, g) = load_param_and_config(model)
    cut_list_entire_model = [] # Add a zero for the first layer (usually a Flatten layer)
    
    if pruned_pairs is None:
        # We just need to create an all zero list for all layers
        for layer in model.layers:
            cut_list_entire_model.append(0)
        return w, g, cut_list_entire_model

    for layer_idx, pairs_at_layer in enumerate(pruned_pairs):
        if len(pairs_at_layer) == 0:
            cut_list_entire_model.append(0)
        else:
            cut_list_curr_layer = []
            for (node_a, node_b) in pairs_at_layer:
                cut_list_curr_layer.append(node_b)
            # We sort in reverse order to cut from larg index to small index, thus after each time we cut
            ##  we do not need to re-index the np arrays.
            cut_list_curr_layer.sort(reverse=True)
            cut_list_entire_model.append(len(cut_list_curr_layer))
            
            size_prev_layer = len(w[layer_idx][0][0])
            assert cut_list_curr_layer[0] < size_prev_layer, "The largest index of hidden unit (" + str(cut_list_curr_layer[0]) + \
                            ") to cut is not smaller than the size of curr layer (" + \
                                str(size_prev_layer) + ")"

            for node in cut_list_curr_layer:
                # Now let's remove the "node_b" hidden unit at the current layer
                ## Cut the connections in the current layer
                list_new_w_layer_idx_0 = []
                try:
                    for index, prev_layer_unit in enumerate(w[layer_idx][0]):
                        list_new_w_layer_idx_0.append(np.delete(prev_layer_unit, node, 0))
                except:
                    print("Something went wrong while cutting FC units at layer", layer_idx) 
                #assert len(w[layer_idx][0]) == len(list_new_w_layer_idx_0), \
                #    "The length of original param at layer " + str(layer_idx) + " should be equal to the newly appened one: " \
                #        + str(len(w[layer_idx][0])) + " vs " + str(len(list_new_w_layer_idx_0))

                w[layer_idx][0] = np.array(list_new_w_layer_idx_0)
                ## Cut the bias in the current layer
                w[layer_idx][1] = np.delete(w[layer_idx][1], node, 0)
                ## Cut the connections in the next layer
                w[layer_idx + 1][0] = np.delete(w[layer_idx + 1][0], node, 0)

    cut_list_entire_model.append(0) # Add a zero for the output layer (usually a Dense layer)
    return w, g, cut_list_entire_model


def create_pruned_model(original_model, pruned_list, path, optimizer=None, loss_fn=None):
    # Let's start building a model
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print("The given path is not available, overwriting the old file ...")

    new_weights, config, cut_list = trim_weights(original_model, pruned_list)
    
    pruned_model = tf.keras.models.Sequential()

    for layer_idx, layer_config in enumerate(config):
        print("Constructing layer", layer_idx)
        if layer_idx == 0:
            if 'flatten' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Flatten(input_shape=config[0]['batch_input_shape'][1:]))
            elif 'conv2d_input' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.InputLayer(input_shape=config[0]['batch_input_shape'][1:]))
            elif 'conv2d' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Conv2D(config[layer_idx]['filters'],
                                                        kernel_size=config[layer_idx]['kernel_size'],
                                                        activation=config[layer_idx]['activation'],
                                                        input_shape=config[0]['batch_input_shape'][1:],
                                                        padding=config[0]['padding'],
                                                        strides=config[0]['strides'],
                                                        trainable=False))
            elif 'dense' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Dense(config[layer_idx]['units'] - cut_list[layer_idx],
                                                   input_shape=config[0]['batch_input_shape'][1:],
                                                   activation=config[layer_idx]['activation'],
                                                   trainable=False))
            else:
                print("Unable to construct layer", layer_idx, "due to incompatible layer type")
        else:
            if 'dense' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Dense(config[layer_idx]['units'] - cut_list[layer_idx],
                                                   activation=config[layer_idx]['activation'],
                                                   trainable=False))
            elif 'conv2d' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Conv2D(config[layer_idx]['filters'],
                                                    kernel_size=config[layer_idx]['kernel_size'],
                                                    activation=config[layer_idx]['activation'],
                                                    padding=config[layer_idx]['padding'],
                                                    strides=config[layer_idx]['strides'],
                                                    trainable=False))
            elif 'flatten' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Flatten())
            elif 'max_pooling2d' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.MaxPooling2D(pool_size=config[layer_idx]['pool_size'],
                                                        strides=config[layer_idx]['strides'],
                                                        trainable=False))
            else:
                print("Unable to construct layer", layer_idx, "due to incompatible layer type")

    if loss_fn is None:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if optimizer is None:
        optimizer = 'adam'

    is_first_layer_input = False 
    if "conv2d_input" in original_model.layers[0].name:
        is_first_layer_input = True

    for index, layer in enumerate(pruned_model.layers):
        if not "dense" in layer.name:
            print("Setting the original weights to layer", index, layer.name)
            if is_first_layer_input:
                layer.set_weights(original_model.layers[index + 1].get_weights())
            else:
                layer.set_weights(original_model.layers[index].get_weights())
            
        else:
            print("Setting the pruned weights to layer", index, layer.name)
            if is_first_layer_input:
                layer.set_weights(new_weights[index + 1])
            else:
                layer.set_weights(new_weights[index])
                    
    pruned_model.compile(optimizer=optimizer,
                         loss=loss_fn,
                         metrics=['accuracy'])
    print(pruned_model.summary())
    pruned_model.save(path)
    return pruned_model
