#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
import progressbar

import paoding.utility.utils as utils
# import matplotlib.gridspec as gridspec

#mpl.rcParams['figure.figsize'] = (7, 7)
#mpl.rcParams['axes.grid'] = False
#mpl.rcParams.update({'font.size': 7})

to_display = False
force_perturbation_even_zero_gradient = True

def decision(probability):
    return random.random() < probability

def load_mnist_handwriting_dataset(normalize=True):
    # The MNIST dataset contains 60,000 28x28 greyscale images of 10 digits.
    # There are 50000 training images and 10000 test images.
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path="mnist.npz")
    # print("Training dataset size: ", train_images.shape, train_labels.shape)
    if normalize:
        # Normalize pixel values to be between 0 and 1
        train_images, test_images = train_images / 255.0, test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)

# Helper function to preprocess the image to a tf.Tensor for the FGSM
def image_preprocess(image_arr):
    image_tensor = tf.cast(image_arr, tf.float32)
    # image = tf.image.resize(image, (28, 28))
    # image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    # image = image[None, ...]
    return image_tensor


# Helper function to post-process the image tensor to an NP array for the plotting purpose
def image_postprocess(image_tensor, dims=[28, 28]):
    image_arr = image_tensor.numpy().reshape(1, dims[0], dims[1])
    return image_arr

# Helper function to extract labels and confidence from a probability vector
def interpret_prediction(probs, top=1):
    if top <= 1:
        label = probs.argmax()
        confidence = probs[label] / sum(probs)
        return label, confidence
    else:
        topk_res = tf.nn.top_k(probs, k=top, sorted=True, name=None)
        label = topk_res.indices.numpy()
        confidence = topk_res.values.numpy()
    return label, confidence

def create_adversarial_pattern(input_image, input_label, pretrained_model, loss_object):
    input_image = image_preprocess(input_image)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    # Return a tensor filled with 0, -1 or +1
    return signed_grad

def create_adversarial_pattern_cifar(input_image, input_label, pretrained_model, loss_object):
    input_image = image_preprocess(input_image)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    # Return a tensor filled with 0, -1 or +1
    return signed_grad

def create_adversarial_pattern_kaggle(input_features, input_label, pretrained_model, loss_object):
    input_features = image_preprocess(input_features)
    with tf.GradientTape() as tape:
        tape.watch(input_features)
        prediction = pretrained_model(input_features)
        loss = loss_object(np.asarray(input_label).astype('float32').reshape((-1,1)), prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_features)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    # Return a tensor filled with 0, -1 or +1
    return signed_grad

def create_one_hot_vector(label, dims=10):
    onehot = tf.one_hot(label, dims)
    return tf.reshape(onehot, (1, dims))

def attack_and_display_images(benign_image, perturbations, adv_image, pretrained_model, description=None, subfig=None, bar_plot=None):
    # _, label, confidence = get_imagenet_label(pretrained_model.predict(image))

    image_arr = image_postprocess(adv_image, dims=[28, 28])
    probs = pretrained_model.predict(image_arr)
    adv_class, confidence = interpret_prediction(probs[0])
    # print(description, probs)
    if not subfig:
        figure, axs = plt.subplots(1,2)
        #plt.figure()
        benign_image_fig = benign_image * 0.5 + 0.5
        adv_image_fig = image_arr[0] * 0.5 + 0.5
        axs[0].imshow(benign_image_fig)
        axs[1].imshow(adv_image_fig)
        plt.title('{} \nLabel {} : {:.2f}% Confidence'.format(description,
                                                         adv_class, confidence * 100))
        plt.show()
        print(probs)
    else:
        subfig.imshow(image_arr[0] * 0.5 + 0.5)
        subfig.set_title('{} \nLabel {} : {:.2f}% Confidence'.format(description,
                                                                adv_class, confidence * 100))
    if bar_plot:
        bars = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        y_pos = np.arange(len(bars))
        bar_plot.set_yticks(y_pos, bars)
        bar_plot.barh(y_pos, probs[0])
    return adv_class


def attack_images(image, pretrained_model):
    # _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
    image_arr = image_postprocess(image, dims=[28, 28])
    probs = pretrained_model.predict(image_arr)
    adv_class, confidence = interpret_prediction(probs[0])
    return adv_class


def attack_and_display_images_cifar(image, pretrained_model, description=None, subfig=None, bar_plot=None):
    # _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
    image_arr = image.numpy().reshape(1, 32, 32, 3)
    probs = pretrained_model.predict(image_arr)
    adv_class, confidence = interpret_prediction(probs[0])
    # print(description, probs)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    if not subfig:
        plt.figure()
        plt.imshow(image_arr[0] * 0.5 + 0.5)
        plt.title('{} \nLabel {}({}) : {:.2f}% Confidence'.format(description,
                                                                class_names[adv_class], adv_class, confidence * 100))

        print(probs)
        plt.show()
    else:
        subfig.imshow(image_arr[0] * 0.5 + 0.5)
        subfig.set_title('{} \nLabel {}({}) : {:.2f}% Confidence'.format(description,
                                                                class_names[adv_class], adv_class, confidence * 100))
    if bar_plot:
        bars = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        y_pos = np.arange(len(bars))
        bar_plot.set_yticks(y_pos, bars)
        bar_plot.barh(y_pos, probs[0])
    return adv_class, confidence

def attack_images_cifar(image, pretrained_model):
    image_arr = image.numpy().reshape(1, 32, 32, 3)
    probs = pretrained_model.predict(image_arr)
    adv_class, confidence = interpret_prediction(probs[0])
    return adv_class, confidence


def attack_images_chest(image, pretrained_model):
    image_arr = image.numpy().reshape(1, 64, 64, 1)
    probs = pretrained_model.predict(image_arr)
    adv_class, confidence = interpret_prediction(probs[0])
    return adv_class, confidence


def attack_features_kaggle(features, pretrained_model):
    features_arr = features.numpy().reshape(1, 29)
    probs = pretrained_model.predict(features_arr)
    adv_class, confidence = interpret_prediction(probs[0])
    return adv_class, confidence


def create_adversarial_example(target_image, perturbations, clip_min=-1, clip_max=1):
    adv_x = target_image + perturbations
    adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
    return adv_x


def robustness_evaluation(model, dataset, epsilons, num_iteration):

    # The MNIST dataset contains 60,000 28x28 greyscale images of 10 digits.
    # There are 50000 training images and 10000 test images.
    test_images, test_labels = dataset

    if  num_iteration == -1 or num_iteration > len(test_images):
        num_iteration = len(test_images)
        print(" >> Number of iteration set to the size of test set, all instances will be tested")

    #################################
    #      PREPARING LOGGING        #
    #################################
    robustness_stat_dict = {}
    for eps in epsilons:
        robustness_stat_dict[eps] = 0

    ################################
    #  CHOOSING A BENIGN INSTANCE  #
    ################################

    # Here we use a fix range of test samples to ensure fairness of experiments
    sample_indexes = list(range(0, num_iteration))

    bar = progressbar.ProgressBar(maxval=num_iteration,
                                  widgets=[progressbar.Bar('=', 'ADVERSARIAL EVALUATION [', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for sample_index in sample_indexes:

        indexes_to_investigae = []
        if sample_index in indexes_to_investigae:
            to_display = True
        else:
            to_display = False

        target_images = test_images[sample_index:sample_index + 1]
        target_labels = test_labels[sample_index:sample_index + 1]
        image_probs = model.predict(target_images)

        #if to_display:
        #    plt.imshow(target_images[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]

        image_class, class_confidence = interpret_prediction(image_probs[0])
        
        # Added on 27 AUG 2022
        # Experimental code: to see if we should only count those correctly classified samples in adversarial assessment.
        #    To this end, we only proceed to the next step if the "image_class" equals to the "target_labels[0]."
        if image_class != target_labels[0]:
            continue
        
        ################################
        #  GENERATING PERTURBATIONS    #
        ################################

        loss_object = tf.keras.losses.CategoricalCrossentropy()

        # Get the input label of the image.
        benign_label = create_one_hot_vector(image_class, dims=image_probs.shape[-1])

        perturbations = create_adversarial_pattern(target_images[0].reshape(1, 784), benign_label, model,
                                                   loss_object)

        # In case there is occurance of ZERO gradient, we xx(randomly)xx -> manually add a sign to the perturbation as either
        #   a + sign or - sign can increase the objective function (loss)
        if not force_perturbation_even_zero_gradient and np.count_nonzero(perturbations) == 0:
            adjusted_perturbations = []
            for sign in perturbations[0]:
                adjusted_perturbations.append(1.0)
                # Remove randomization to ensure consistency of cross device experiment
                #if decision(0.5):
                #    adjusted_perturbations.append(1.0)
                #else:
                #    adjusted_perturbations.append(-1.0)
            perturbations = tf.sign(tf.reshape(tf.Variable(adjusted_perturbations), perturbations.shape))

        if force_perturbation_even_zero_gradient:
            adjusted_perturbations = []
            for sign in tf.reshape(perturbations[0],(-1)):
                if sign == -1:
                    adjusted_perturbations.append(-1.0)
                else:
                    adjusted_perturbations.append(1.0)
            perturbations = tf.sign(adjusted_perturbations)

        ################################
        #    GENERATING ADV SAMPLES    #
        ################################
        # Let's try this out for different values of epsilon and observe the resultant image
        # Perturbations here are just sign made up with 0, -1 and +1, an epsilon multiplier is needed.
        perturbations = tf.reshape(perturbations, [28, 28])

        for i, eps in enumerate(epsilons):
            perts = eps * perturbations
            adv_x = create_adversarial_example(target_images[0], perts)
            if to_display:
                descriptions = 'Epsilon = {'+str(eps)+'}'
                adv_class = attack_and_display_images(target_images[0], perturbations, adv_x, model, description=descriptions)
            else:
                adv_class = attack_images(adv_x, model)
            # Record the maximum epsilon that the classifier still not to misbehave
            if image_class == adv_class:
                robustness_stat_dict[eps] = robustness_stat_dict[eps] + 1
            #else:
                #print("#",sample_index,"; original label:",image_class,"; attack label:",adv_class)
                #print(str(sample_index), end=" ")
        bar.update(sample_index)
    bar.finish()

    return robustness_stat_dict


def robustness_evaluation_cifar(model, dataset, epsilons, num_iteration):

    # The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes.
    # There are 50000 training images and 10000 test images.
    test_images, test_labels = dataset

    if  num_iteration == -1 or num_iteration > len(test_images):
        num_iteration = len(test_images)
        print(" >> Number of iteration set to the size of test set, all instances will be tested")

    #################################
    #      PREPARING LOGGING        #
    #################################
    robustness_stat_dict = {}
    for eps in epsilons:
        robustness_stat_dict[eps] = 0

    ################################
    #  CHOOSING A BENIGN INSTANCE  #
    ################################

    # Here we use a fix range of test samples to ensure fairness of experiments
    sample_indexes = list(range(0, num_iteration))

    bar = progressbar.ProgressBar(maxval=num_iteration,
                                  widgets=[progressbar.Bar('=', 'ADVERSARIAL EVALUATION [', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for sample_index in sample_indexes:

        indexes_to_investigae = []
        if sample_index in indexes_to_investigae:
            to_display = True
        else:
            to_display = False

        target_images = test_images[sample_index:sample_index + 1]
        target_labels = test_labels[sample_index:sample_index + 1]
        image_probs = model.predict(target_images)

        #plt.imshow(target_images[0])
        #plt.show()
        image_class, class_confidence = interpret_prediction(image_probs[0])

        # Added on 27 AUG 2022
        # Experimental code: to see if we should only count those correctly classified samples in adversarial assessment.
        #    To this end, we only proceed to the next step if the "image_class" equals to the "target_labels[0]."
        if image_class != target_labels[0]:
            continue

        ################################
        #  GENERATING PERTURBATIONS    #
        ################################

        loss_object = tf.keras.losses.CategoricalCrossentropy()

        # Get the input label of the image.
        benign_label = create_one_hot_vector(image_class, dims=image_probs.shape[-1])
        benign_label = tf.reshape(benign_label, (1, image_probs.shape[-1]))
        perturbations = create_adversarial_pattern_cifar(target_images, benign_label, model,
                                                   loss_object)

        # In case there is occurance of ZERO gradient, we xx(randomly)xx -> manually add a sign to the perturbation as either
        #   a + sign or - sign can increase the objective function (loss)
        if not force_perturbation_even_zero_gradient and np.count_nonzero(perturbations) == 0:
            adjusted_perturbations = []
            for sign in tf.reshape(perturbations[0],(-1)):
                adjusted_perturbations.append(1.0)
                # Remove randomization to ensure consistency of cross device experiment
                #if decision(0.5):
                #    adjusted_perturbations.append(1.0)
                #else:
                #    adjusted_perturbations.append(-1.0)
            perturbations = tf.sign(adjusted_perturbations)

        if force_perturbation_even_zero_gradient:
            adjusted_perturbations = []
            for sign in tf.reshape(perturbations[0],(-1)):
                if sign == -1:
                    adjusted_perturbations.append(-1.0)
                else:
                    adjusted_perturbations.append(1.0)
            perturbations = tf.sign(adjusted_perturbations)

        ################################
        #    GENERATING ADV SAMPLES    #
        ################################
        # Let's try this out for different values of epsilon and observe the resultant image
        # Perturbations here are just sign made up with 0, -1 and +1, an epsilon multiplier is needed.
        perturbations = tf.reshape(perturbations, target_images[0].shape)

        for i, eps in enumerate(epsilons):
            perts = eps * perturbations
            adv_x = create_adversarial_example(target_images[0], perts)
            if to_display:
                descriptions = 'Epsilon = {' + str(eps) + '}'
                adv_class, adv_confidence = attack_and_display_images_cifar(adv_x, model, description=descriptions)
            else:
                adv_class, adv_confidence = attack_images_cifar(adv_x, model)
            # Record the maximum epsilon that the classifier still not to misbehave
            if image_class == adv_class:
                robustness_stat_dict[eps] = robustness_stat_dict[eps] + 1
            #else:
                #print("#",sample_index,"; original label:",image_class,"; attack label:",adv_class)
                #print(str(sample_index), end=" ")
        bar.update(sample_index)
    bar.finish()

    return robustness_stat_dict


def robustness_evaluation_cifar_topK(model, dataset, epsilons, num_iteration, k):

    # The CIFAR-100 dataset contains 60,000 32x32 color images in 100 classes.
    # There are 50000 training images and 10000 test images.
    test_images, test_labels = dataset

    if  num_iteration == -1 or num_iteration > len(test_images):
        num_iteration = len(test_images)
        print(" >> Number of iteration set to the size of test set, all instances will be tested")

    #################################
    #      PREPARING LOGGING        #
    #################################
    robustness_stat_dict = {}
    for eps in epsilons:
        robustness_stat_dict[eps] = 0

    ################################
    #  CHOOSING A BENIGN INSTANCE  #
    ################################

    # Here we use a fix range of test samples to ensure fairness of experiments
    sample_indexes = list(range(0, num_iteration))

    bar = progressbar.ProgressBar(maxval=num_iteration,
                                  widgets=[progressbar.Bar('=', 'ADVERSARIAL EVALUATION [', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for sample_index in sample_indexes:

        indexes_to_investigae = []
        if sample_index in indexes_to_investigae:
            to_display = True
        else:
            to_display = False

        target_images = test_images[sample_index:sample_index + 1]
        target_labels = test_labels[sample_index:sample_index + 1]
        image_probs = model.predict(target_images)

        #plt.imshow(target_images[0])
        #plt.show()
        image_classes, class_confidences = interpret_prediction(image_probs[0], top=k)

        # Added on 27 AUG 2022
        # Experimental code: to see if we should only count those correctly classified samples in adversarial assessment.
        #    To this end, we only proceed to the next step if the "image_class" equals to the "target_labels[0]."
        if target_labels[0] not in image_classes:
            continue

        ################################
        #  GENERATING PERTURBATIONS    #
        ################################

        loss_object = tf.keras.losses.CategoricalCrossentropy()

        # Get the input label of the image.
        benign_label = create_one_hot_vector(target_labels[0], dims=image_probs.shape[-1])
        benign_label = tf.reshape(benign_label, (1, image_probs.shape[-1]))
        perturbations = create_adversarial_pattern_cifar(target_images, benign_label, model,
                                                   loss_object)

        # In case there is occurance of ZERO gradient, we xx(randomly)xx -> manually add a sign to the perturbation as either
        #   a + sign or - sign can increase the objective function (loss)
        if not force_perturbation_even_zero_gradient and np.count_nonzero(perturbations) == 0:
            adjusted_perturbations = []
            for sign in tf.reshape(perturbations[0],(-1)):
                adjusted_perturbations.append(1.0)
                # Remove randomization to ensure consistency of cross device experiment
                #if decision(0.5):
                #    adjusted_perturbations.append(1.0)
                #else:
                #    adjusted_perturbations.append(-1.0)
            perturbations = tf.sign(adjusted_perturbations)

        if force_perturbation_even_zero_gradient:
            adjusted_perturbations = []
            for sign in tf.reshape(perturbations[0],(-1)):
                if sign == -1:
                    adjusted_perturbations.append(-1.0)
                else:
                    adjusted_perturbations.append(1.0)
            perturbations = tf.sign(adjusted_perturbations)

        ################################
        #    GENERATING ADV SAMPLES    #
        ################################
        # Let's try this out for different values of epsilon and observe the resultant image
        # Perturbations here are just sign made up with 0, -1 and +1, an epsilon multiplier is needed.
        perturbations = tf.reshape(perturbations, target_images[0].shape)

        for i, eps in enumerate(epsilons):
            perts = eps * perturbations
            adv_x = create_adversarial_example(target_images[0], perts)
            if to_display:
                descriptions = 'Epsilon = {' + str(eps) + '}'
                adv_class, adv_confidence = attack_and_display_images_cifar(adv_x, model, description=descriptions)
            else:
                adv_class, adv_confidence = attack_images_cifar(adv_x, model)
            # Record the maximum epsilon that the classifier still not to misbehave (still within the top K prediction)
            if adv_class in image_classes:
                robustness_stat_dict[eps] = robustness_stat_dict[eps] + 1
            #else:
                #print("#",sample_index,"; original label:",target_labels[0],"; attack label:",adv_class, " -- not in top K result", image_classes)
                #print(str(sample_index), end=" ")
        bar.update(sample_index)
    bar.finish()

    return robustness_stat_dict


def robustness_evaluation_chest(model, dataset, epsilons, num_iteration):

    # The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes.
    # There are 50000 training images and 10000 test images.
    test_images, test_labels = dataset

    if  num_iteration == -1 or num_iteration > len(test_images):
        num_iteration = len(test_images)
        print(" >> Number of iteration set to the size of test set, all instances will be tested")

    #################################
    #      PREPARING LOGGING        #
    #################################
    robustness_stat_dict = {}
    for eps in epsilons:
        robustness_stat_dict[eps] = 0

    ################################
    #  CHOOSING A BENIGN INSTANCE  #
    ################################
    sample_indexes = random.sample(range(0, len(test_images)), num_iteration)
    #sample_indexes = list(range(0, num_iteration))

    bar = progressbar.ProgressBar(maxval=num_iteration,
                                  widgets=[progressbar.Bar('=', 'ADVERSARIAL EVALUATION [', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for count, sample_index in enumerate(sample_indexes):

        target_images = test_images[sample_index:sample_index + 1]
        target_labels = test_labels[sample_index:sample_index + 1]
        image_probs = model.predict(target_images)

        image_class, class_confidence = interpret_prediction(image_probs[0])
        '''
        if image_class != target_labels[0]:
            class_names = ['PNEUMONIA', 'NORMAL']
            plt.imshow(target_images[0], cmap=plt.cm.binary)
            plt.title('Instance #{} exp. label {} but classified as {}'.format(sample_index,
                                                                               class_names[target_labels[0]],
                                                                               class_names[image_class]))
            plt.show()
        '''
        
        # Added on 27 AUG 2022
        # Experimental code: to see if we should only count those correctly classified samples in adversarial assessment.
        #    To this end, we only proceed to the next step if the "image_class" equals to the "target_labels[0]."
        if image_class != target_labels[0]:
            continue

        ################################
        #  GENERATING PERTURBATIONS    #
        ################################

        loss_object = tf.keras.losses.CategoricalCrossentropy()

        # Get the input label of the image.
        benign_label = create_one_hot_vector(image_class, dims=image_probs.shape[-1])
        benign_label = tf.reshape(benign_label, (1, image_probs.shape[-1]))
        perturbations = create_adversarial_pattern_cifar(target_images, benign_label, model,
                                                   loss_object)

        # In case there is occurance of ZERO gradient, we xx(randomly)xx -> manually add a sign to the perturbation as either
        #   a + sign or - sign can increase the objective function (loss)
        if np.count_nonzero(perturbations) == 0:
            adjusted_perturbations = []
            for sign in tf.reshape(perturbations[0],(-1)):
                adjusted_perturbations.append(1.0)
                # Remove randomization to ensure consistency of cross device experiment
                #if decision(0.5):
                #    adjusted_perturbations.append(1.0)
                #else:
                #    adjusted_perturbations.append(-1.0)
            perturbations = tf.sign(adjusted_perturbations)
        ################################
        #    GENERATING ADV SAMPLES    #
        ################################
        # Let's try this out for different values of epsilon and observe the resultant image
        # Perturbations here are just sign made up with 0, -1 and +1, an epsilon multiplier is needed.
        perturbations = tf.reshape(perturbations, target_images[0].shape)

        for i, eps in enumerate(epsilons):
            perts = eps * perturbations
            adv_x = create_adversarial_example(target_images[0], perts)
            adv_class, adv_confidence = attack_images_chest(adv_x, model)
            #if image_class == 1:
                #gs = gridspec.GridSpec(1, 6)
                #fig = plt.figure()
                # Keep some sub figures at the first row blanks
                #ax1 = fig.add_subplot(gs[0, 1:3])
                #ax2 = fig.add_subplot(gs[0, 3:5])
                #ax1.imshow(target_images[0])
                #ax1.set_title('Instance #{}Label {} - {:.2f}% Confidence'.format(sample_index, image_class, class_confidence * 100))
                #ax2.set_title(
                #    'Label {} - {:.2f}% Confidence'.format(adv_class, adv_confidence * 100))
                #ax2.imshow(adv_x)
                #plt.show()

            # Record the maximum epsilon that the classifier still not to misbehave
            if image_class == adv_class:
                robustness_stat_dict[eps] = robustness_stat_dict[eps] + 1
            # else:
                # print("Adv class is", adv_class, "but expected class is", image_class)

        bar.update(count)
    bar.finish()

    return robustness_stat_dict


def robustness_evaluation_kaggle(model, dataset, epsilons, num_iteration):

    # The kaggle credit card dataset contains 284,807 data with 29 features and 1 boolean label (0 or 1).
    # There are 80% training images and 20% test images.
    test_features, test_labels = dataset

    if  num_iteration == -1 or num_iteration > len(test_features):
        num_iteration = len(test_features)
        print(" >> Number of iteration set to the size of test set, all instances will be tested")

    #################################
    #      PREPARING LOGGING        #
    #################################
    robustness_stat_dict = {}
    for eps in epsilons:
        robustness_stat_dict[eps] = 0

    ################################
    #  CHOOSING A BENIGN INSTANCE  #
    ################################
    
    # Here we use a fix range of test samples to ensure fairness of experiments
    sample_indexes = list(range(0, num_iteration))

    bar = progressbar.ProgressBar(maxval=num_iteration,
                                  widgets=[progressbar.Bar('=', 'ADVERSARIAL EVALUATION [', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for sample_index in sample_indexes:

        target_features = test_features[sample_index:sample_index + 1]
        target_labels = test_labels[sample_index:sample_index + 1]
        image_probs = model.predict(target_features)

        #plt.imshow(target_images[0])
        #plt.show()
        benign_label, class_confidence = interpret_prediction(image_probs[0])

        # Added on 27 AUG 2022
        # Experimental code: to see if we should only count those correctly classified samples in adversarial assessment.
        #    To this end, we only proceed to the next step if the "image_class" equals to the "target_labels[0]."
        if benign_label != target_labels[0]:
            continue

        ################################
        #  GENERATING PERTURBATIONS    #
        ################################

        loss_object = tf.keras.losses.BinaryCrossentropy()

        # Get the input label of the input.
        #benign_label = tf.reshape(benign_label, (1, image_probs.shape[-1]))
        perturbations = create_adversarial_pattern_kaggle(target_features, benign_label, model,
                                                   loss_object)

        # In case there is occurance of ZERO gradient, we xx(randomly)xx -> manually add a sign to the perturbation as either
        #   a + sign or - sign can increase the objective function (loss)
        if np.count_nonzero(perturbations) == 0:
            adjusted_perturbations = []
            for sign in tf.reshape(perturbations[0],(-1)):
                adjusted_perturbations.append(1.0)
                # Remove randomization to ensure consistency of cross device experiment
                #if decision(0.5):
                #    adjusted_perturbations.append(1.0)
                #else:
                #    adjusted_perturbations.append(-1.0)
            perturbations = tf.sign(adjusted_perturbations)
        ################################
        #    GENERATING ADV SAMPLES    #
        ################################
        # Let's try this out for different values of epsilon and observe the resultant image
        # Perturbations here are just sign made up with 0, -1 and +1, an epsilon multiplier is needed.
        perturbations = tf.reshape(perturbations, target_features[0].shape)

        for i, eps in enumerate(epsilons):
            # The definition range of input feature is [-5,5]
            perts = eps * perturbations * 10
            adv_x = create_adversarial_example(target_features[0], perts, clip_min=-5, clip_max=5)
            adv_class, adv_confidence = attack_features_kaggle(adv_x, model)
            # Record the maximum epsilon that the classifier still not to misbehave
            if benign_label == adv_class:
                robustness_stat_dict[eps] = robustness_stat_dict[eps] + 1

        bar.update(sample_index)
    bar.finish()

    return robustness_stat_dict
