import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import os
import math
import kerassurgeon.identify
from kerassurgeon.operations import delete_channels, delete_layer
from sklearn import cluster
import numpy as np
import sklearn
from keras.models import Model
from kerassurgeon.surgeon import Surgeon

#test

#function to return pruned filters with l1 method
def prune_l1(model, n_pruned, layer=None):
    """returns list of indexes of filter to prune or a matrix layer X filter to prune"""
    if layer or layer==0:
        norms = get_filters_l1(model,layer)
        to_prune = np.argsort(norms)[:n_pruned]
    
    else:
        norms = get_filters_l1(model)
        to_prune = smallest_indices(norms, n_pruned)
    
    return to_prune

#function to return pruned filters with apoz method
def prune_apoz(model, n_pruned, layer=None):
    """returns list of indexes of filter to prune or a matrix layer X filter to prune"""
    if layer or layer==0:
        apoz = get_filters_apoz(model,layer)
        to_prune = np.argsort(apoz)[::-1][:n_pruned]
    
    else:
        apoz = get_filters_apoz(model)
        to_prune = biggest_indices(apoz, n_pruned)
    
    return to_prune

def prune_random(model, n_pruned, layer=None):
    """returns list of indexes of filter to prune or a matrix layer X filter to prune"""
    weights = get_filter_weights(model, layer)
    if layer or layer==0:
        n_filters = weights.shape[3]
        to_prune = np.random.choice(range(n_filters), n_pruned, replace=False)
    else:
        layer_ix = np.random.choice(len(weights))
        filters = weights[layer_ix].shape[3]
        filter_ix = np.random.choice(range(filters))
        to_prune = [[layer_ix, filter_ix]]

        for i in range(n_pruned-1):
            while [layer_ix, filter_ix] in to_prune :
                #choose layer
                layer_ix = np.random.choice(len(weights))
                #choose filter 
                filters = weights[layer_ix].shape[3]
                filter_ix = np.random.choice(range(filters))
            to_prune.append([layer_ix, filter_ix])

        to_prune = np.array(to_prune)
    return to_prune

#function to return pruned filters with K-means method
def prune_kmeans(model, n_pruned, method='kmeans_random', layer=None):
    """
    Use Kmeans to prune filters based on similarity
    Arguments :
        model: A Keras Model
        n_pruned : number of filters to prune
        method : Pruning method :
                    - kmeans_random : removes one random element per cluster
                    - kmeans_centroid : keeps only filter closest to centroid
    Returns :
        a list of filter indexes to prune
    
    """
    assert layer is not None, "Layer index should be specified, Kmeans pruning only works on a layer"
    assert method in ['kmeans_random', 'kmeans_centroid'], "Method should be in kmeans_random or kmeans_centroid"
    
    if method == 'kmeans_random':
        pruned_indexes = kmeans_random(model, n_pruned, layer)
            
    if method == 'kmeans_centroid':
        pruned_indexes = kmeans_centroid(model, n_pruned, layer)
    
    return pruned_indexes

def kmeans_random(model, n_pruned, layer):
    """Fit a K means with k=n_pruned and remove randomly one filter per cluster"""
    weights = get_filter_weights(model, layer)
    num_filter = len(weights[0,0,0,:])
    filter_array = weights.T.reshape(num_filter, 
                                     weights.shape[0]*weights.shape[1]*weights.shape[2])
    
    kmeans = sklearn.cluster.KMeans(n_clusters=n_pruned)
    kmeans.fit(filter_array)
    cluster_indexes = kmeans.labels_
    to_prune = []
    for i in range(n_pruned):
        cluster_filters = np.where(cluster_indexes==i)[0]
        filter_index = np.random.choice(cluster_filters)
        to_prune.append(filter_index)
    
    return to_prune

def kmeans_centroid(model, n_pruned, layer):
    """Fit a Kmeans with k=total_filters -n_pruned and keeps the filters closest to the centroids"""
    weights = get_filter_weights(model, layer)
    num_filter = len(weights[0,0,0,:])
    filter_array = weights.T.reshape(num_filter, 
                                     weights.shape[0]*weights.shape[1]*weights.shape[2])
    n_clusters = num_filter - n_pruned
    
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(filter_array)
    centroids = kmeans.cluster_centers_
    cluster_indexes = kmeans.labels_

    to_prune = []
    for i in range(n_clusters):
        cluster_filters = np.where(cluster_indexes==i)[0]
        centroid = centroids[i]
        distances = [np.linalg.norm(filter_array[ix]-centroid) for ix in cluster_filters]
        closest_ix = np.argmin(distances)
        to_keep = cluster_filters[closest_ix]
        to_prune.extend([el for el in cluster_filters if el != to_keep])
    
    return to_prune

def get_filter_weights(model, layer=None):
    """function to return weights array for one or all layers"""
    if layer or layer==0:
        weight_array = model.layers[layer].get_weights()[0]
        
    else:
        # Mark's modification on 14 Jul
        '''
        weights = [model.layers[layer_ix].get_weights()[0] for layer_ix in range(len(model.layers))\
            if 'conv' in model.layers[layer_ix].name and 'conv2d_input' not in model.layers[layer_ix].name]
        '''
        weights = []
        for layer_ix in range(len(model.layers)):
            if 'conv' in model.layers[layer_ix].name and 'conv2d_input' not in model.layers[layer_ix].name:
                if model.layers[layer_ix].get_weights():
                    weights.append(model.layers[layer_ix].get_weights()[0])
        weight_array = [np.array(i) for i in weights]
    
    return weight_array

def get_filters_l1(model, layer=None):
    """Returns L1 norm of a Keras model filters at a given layer, if layer=None, returns a matrix of norms"""
    if layer or layer==0:
        weights = get_filter_weights(model, layer)
        num_filter = len(weights[0,0,0,:])
        norms_dict = {}
        norms = []
        for i in range(num_filter):
            l1_norm = np.sum(abs(weights[:,:,:,i]))
            norms.append(l1_norm)
    else:
        weights = get_filter_weights(model)
        max_kernels = max([layr.shape[3] for layr in weights])
        norms = np.empty((len(weights), max_kernels))
        norms[:] = np.NaN
        for layer_ix in range(len(weights)):
            # compute norm of the filters
            kernel_size = weights[layer_ix][:,:,:,0].size
            nb_filters = weights[layer_ix].shape[3]
            kernels = weights[layer_ix]
            l1 = [np.sum(abs(kernels[:,:,:,i])) for i in range(nb_filters)]
            # divide by shape of the filters
            l1 = np.array(l1) / kernel_size
            norms[layer_ix, :nb_filters] = l1
    return norms
    

def get_filters_apoz(model, layer=None):
    
    # Get a sample of the train set , or should it be the validation set ?
    test_generator = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    apoz_dir = "/home/ec2-user/Telecom/experiments/data/imagenette2-320/train"

    apoz_generator = test_generator.flow_from_directory(
                apoz_dir,
                target_size=(160, 160),
                batch_size=1,
                class_mode='categorical',
                subset='validation',
                shuffle= False)
    
    if layer or layer ==0:
        assert 'conv' in model.layers[layer].name, "The layer provided is not a convolution layer"
        weights_array = get_filter_weights(model, layer)
        act_ix = layer + 1
        nb_filters = weights_array.shape[3]
        apoz = compute_apoz(model, act_ix, nb_filters, apoz_generator)
                
    else :
        weights_array = get_filter_weights(model)
        max_kernels = max([layr.shape[3] for layr in weights_array])

        conv_indexes = [i for i, v in enumerate(model.layers) if 'conv' in v.name]
        activations_indexes = [i for i,v in enumerate(model.layers) if 'activation' \
                       in v.name and 'conv' in model.layers[i-1].name]

        # create nd array to collect values
        apoz = np.zeros((len(weights_array), max_kernels))

        for i, act_ix in enumerate(activations_indexes):
            # score this sample with our model (trimmed to the layer of interest)
            nb_filters = weights_array[i].shape[3]
            apoz_layer = compute_apoz(model, act_ix, nb_filters, apoz_generator)
            apoz[i, :nb_filters] = apoz_layer
        
    return apoz

def compute_apoz(model, layer_ix, nb_filters, generator):
    """Compute Average percentage of zeros over a layers activation maps"""
    act_layer = model.get_layer(index=layer_ix)
    node_index = 0
    temp_model = Model(model.inputs,
                               act_layer.get_output_at(node_index)
                              )


            # count the percentage of zeros per activation
    a = temp_model.predict_generator(generator,944, workers=3, verbose=1)
    activations = a.reshape(a.shape[0]*a.shape[1]*a.shape[2],nb_filters).T
    apoz_layer = np.sum(activations == 0, axis=1) / activations.shape[1]
    
    return apoz_layer

def prune_model(model, perc, opt, method='l1', layer=None):
    """Prune a Keras model using different methods
    Arguments:
        model: Keras Model object
        perc: a float between 0 and 1
        method: method to prune, can be one of ['l1','apoz','kmeans_random','kmeans_centroid','random']
    Returns:
        A pruned Keras Model object
    
    """
    assert method in ['l1','apoz','kmeans_random','kmeans_centroid','random'], "Invalid pruning method"
    assert perc >=0 and perc <1, "Invalid pruning percentage"
    
    
    #n_pruned = compute_pruned_count(model, perc, layer)
    # Mannually set number to be pruned to 1
    n_pruned = 1

    if method =='l1':
        to_prune = prune_l1(model, n_pruned, layer)    
    if method =='apoz':
        to_prune = prune_apoz(model, n_pruned, layer)
    if method =='random':
        to_prune = prune_random(model, n_pruned, layer)
    if 'kmeans' in method:
        assert layer is not None, "Kmeans based pruning requires to specify layer"
        to_prune = prune_kmeans(model, n_pruned, method, layer)
    
    print(" +++ FINISHED SAMPLING")

    if layer or layer ==0:
        model_pruned = prune_one_layer(model, to_prune, layer, opt)
    else:
        model_pruned = prune_multiple_layers(model, to_prune, opt)
            
    return model_pruned

def prune_one_layer(model, pruned_indexes, layer_ix, opt):
    """Prunes one layer based on a Keras Model, layer index and indexes of filters to prune"""
    model_pruned = delete_channels(model, model.layers[layer_ix], pruned_indexes)
    model_pruned.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
    return model_pruned

def prune_multiple_layers(model, pruned_matrix, opt):
    conv_indexes = [i for i, v in enumerate(model.layers) if 'conv' in v.name and 'input' not in v.name]
    layers_to_prune = np.unique(pruned_matrix[:,0])
    surgeon = Surgeon(model, copy=True)
    to_prune = pruned_matrix
    to_prune[:,0] = np.array([conv_indexes[i] for i in to_prune[:,0]])
    layers_to_prune = np.unique(to_prune[:,0])
    for layer_ix in layers_to_prune :
        print(" +++ PRUNING LAYER", layer_ix, model.layers[layer_ix].name)
        pruned_filters = [x[1] for x in to_prune if x[0]==layer_ix]
        pruned_layer = model.layers[layer_ix]
        surgeon.add_job('delete_channels', pruned_layer, channels=pruned_filters)
    
    model_pruned = surgeon.operate()
    model_pruned.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    
    return model_pruned

def compute_pruned_count(model, perc, layer=None):
    if layer or layer ==0:
            # count nb of filters
        nb_filters = model.layers[layer].output_shape[3]
    else:
        nb_filters = 0
        for i, layer in enumerate(model.layers):
            # print(" XXX ", i, layer, model.layers[i].name)
            # Mark's modification on 14 Jul
            if 'conv' in model.layers[i].name and not 'conv2d_input' in model.layers[i].name:
                #print(model.layers[i].name)
                #print(model.layers[i].output_shape)
                #print(model.layers[i].output_shape[3])
                nb_filters += model.layers[i].output_shape[3]
        # nb_filters = np.sum([model.layers[i].output_shape[3] for i, layer in enumerate(model.layers) if 'conv' in model.layers[i].name])
            
    n_pruned = int(np.ceil(perc*nb_filters))
    print(" +++", n_pruned, "filters to be pruned")
    return n_pruned

#function to evaluate model on validation set
def evaluate(model, validation_generator):
    filenames = validation_generator.filenames
    nb_samples = len(filenames)
    evaluate = model.evaluate_generator(validation_generator,nb_samples, workers=3, verbose=1)
    accuracy = evaluate[1]
    return accuracy

# get model size, nb of weights, accuracy, FLOPS
def get_model_info(model_path, model):
    size = os.path.getsize(model_path)
    flops = get_flops(model)
    weights = model.count_params()
    return {'size':size, 'flops':flops, 'total_weights':weights}

# get model flops
def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, 
                                cmd='op', 
                                options=opts)

    return flops.total_float_ops

#find n smallest indices in numpy ndarray
def smallest_indices(array, N):
    idx = array.ravel().argsort()[:N]
    return np.stack(np.unravel_index(idx, array.shape)).T

def biggest_indices(array, N):
    idx = array.ravel().argsort()[::-1][:N]
    return np.stack(np.unravel_index(idx, array.shape)).T