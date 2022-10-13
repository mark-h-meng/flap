#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
#import matplotlib.pyplot as plt
import os, shutil
import numpy as np
#from sklearn.metrics import confusion_matrix
#import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
# VGG19 Model 
from tensorflow.keras.applications.vgg19 import VGG19

# Quick fix to encounter memory growth issue when working on shared GPU workstation
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def read_pneumonia_images_from_local(data_path, img_size=64):
    labels = ['PNEUMONIA', 'NORMAL']
    data = []
    for label in labels:
        path = os.path.join(data_path, label).replace("\\", "/")
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img).replace("\\", "/")
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


def load_data_pneumonia(data_path, img_size = 64):

    train = read_pneumonia_images_from_local(data_path + '/train', img_size)
    test = read_pneumonia_images_from_local(data_path + '/test', img_size)
    val = read_pneumonia_images_from_local(data_path + '/val', img_size)

    x_train = []
    y_train = []

    x_val = []
    y_val = []

    x_test = []
    y_test = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    # Normalize the data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255
    x_test = np.array(x_test) / 255

    # resize data for deep learning
    x_train = x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)

    x_val = x_val.reshape(-1, img_size, img_size, 1)
    y_val = np.array(y_val)

    x_test = x_test.reshape(-1, img_size, img_size, 1)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


def load_data_creditcard_from_csv(data_path):
    if os.path.exists(data_path):
        print(">> Local data file found", data_path)
        raw_df = pd.read_csv(data_path)
    else:
        print(">> Local data file not found, reading the file from Internet ...")
        raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')

    ## Examine the class label imbalance
    neg, pos = np.bincount(raw_df['Class'])

    ## Clean, split and normalize the data
    cleaned_df = raw_df.copy()

    # You don't want the `Time` column.
    cleaned_df.pop('Time')

    # The `Amount` column covers a huge range. Convert to log-space.
    eps = 0.001  # 0 => 0.1¢
    cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount') + eps)

    # Use a utility from sklearn to split and shuffle our dataset.
    #  Test set: 20%, Val. set: 16%, Training set: 64%
    train_df, test_df = train_test_split(cleaned_df, test_size=0.2)

    # Separate samples and labels
    train_labels = np.array(train_df.pop('Class'))
    test_labels = np.array(test_df.pop('Class'))

    train_features = np.array(train_df)
    test_features = np.array(test_df)

    # Normalize the input features using the sklearn StandardScaler. This will set the mean to 0 and standard deviation to 1.
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.fit_transform(test_features)

    train_features = np.clip(train_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)
    return (train_features, train_labels), (test_features, test_labels)


def train_creditcard_3_layer_mlp(train_data, test_data, path, overwrite=False,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                            epochs=100, initial_bias=[-6.35935934]):
    BATCH_SIZE = 2048

    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]

    (x, y)=train_data
    (test_features, test_labels)=test_data
    train_features, val_features, train_labels, val_labels = train_test_split(x, y,
                                                                              test_size=0.2, train_size=0.8)

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print("Model not found, create and train a new one ...")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)

        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(train_features.shape[-1],)))
        #model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))
        #model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid',
                               bias_initializer=tf.keras.initializers.Constant(initial_bias)))

        print(model.summary())
        model.compile(optimizer=optimizer_config, loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=METRICS)

        training_history = model.fit(train_features, train_labels,
                                     batch_size=BATCH_SIZE,
                                     epochs=epochs,
                                     callbacks=[early_stopping],
                                     validation_data=(val_features, val_labels))

        baseline_results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
        test_loss, test_accuracy = baseline_results[0], baseline_results[-4]

        test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
        
        print("Final Accuracy achieved is: ", test_accuracy, "with Loss", test_loss)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")
        #plt.show()

    else:
        print("Model found, there is no need to re-train the model ...")


def train_cifar_8_layer_cnn(train_data,
                            test_data,
                            path,
                            overwrite=False,
                            use_relu=False,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            epochs=50):

    (train_images, train_labels)=train_data
    (test_images, test_labels)=test_data

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print(path, " - model not found, create and train a new one ...")
        model = models.Sequential()
        # In the first layer, please provide the input shape (32,32,3)
        model.add(
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # Result will be 32 30*30 outputs
        model.add(layers.MaxPooling2D((2, 2)))  # Result will be 32 15*15 outputs
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Result will be 64 13*13 outputs
        model.add(layers.MaxPooling2D((2, 2)))  # Result will be 64 6*6 outputs
        model.add(layers.Conv2D(64, (3, 3), activation='sigmoid'))  # Result will be 64 4*4 outputs

        model.add(layers.Flatten())  # Result will be a vector with length 4*4*64 = 1024
        if use_relu:
            model.add(layers.Dense(64, activation='relu'))  # Result will 64 outputs
        else:
            model.add(layers.Dense(64, activation='sigmoid'))  # Result will 64 outputs

        model.add(layers.Dense(10, activation='softmax'))  # Result will be 10 outputs

        print(model.summary())
        model.compile(optimizer=optimizer_config, loss=loss_fn,
                      metrics=['accuracy'])

        training_history = model.fit(train_images, train_labels, epochs=epochs,
                                     validation_data=(test_images, test_labels))

        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
        print("Final Accuracy achieved is: ", test_accuracy)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")
        '''
        plt.plot(training_history.history['accuracy'], label="Accuracy")
        plt.plot(training_history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.4, 1])
        plt.legend(loc='lower right')
        #plt.show()
        '''
    else:
        print("Model found, there is no need to re-train the model ...")

def train_cifar_cnn(train_data,
                            test_data,
                            path,
                            overwrite=False,
                            use_relu=False,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            epochs=20,
                            topK=1):

    (x, y)=train_data
    (test_images, test_labels)=test_data

    train_images, val_images , train_labels, val_labels = train_test_split(x, y, test_size=0.167, train_size=0.833)

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print(path, " - model not found, create and train a new one ...")
        model = models.Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(32, 32, 3)))
        model.add(layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2))) 

        model.add(layers.Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
            
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        print(model.summary())
        
        if topK <= 1:
            model.compile(optimizer=optimizer_config, loss=loss_fn,
                      metrics=['accuracy'])
            
            training_history = model.fit(train_images, train_labels, epochs=epochs,
                                        validation_data=(val_images, val_labels))

            test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
            print("Final Accuracy achieved is: ", test_accuracy)

        else:
            model.compile(optimizer=optimizer_config, 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=topK)])
            
            training_history = model.fit(train_images, train_labels, epochs=epochs,
                                        validation_data=(val_images, val_labels))

            test_loss, test_accuracy, test_topk_accuracy = model.evaluate(test_images, test_labels, verbose=2)
            print("Final Accuracy achieved is: ", test_accuracy, " with top-K accuracy as", test_topk_accuracy)
        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")

    else:
        print("Model found, there is no need to re-train the model ...")


def train_cifar_9_layer_cnn(train_data,
                            test_data,
                            path,
                            overwrite=False,
                            use_relu=False,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            epochs=20,
                            topK=1):

    (x, y)=train_data
    (test_images, test_labels)=test_data

    train_images, val_images , train_labels, val_labels = train_test_split(x, y, test_size=0.167, train_size=0.833)

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print(path, " - model not found, create and train a new one ...")
        model = models.Sequential()
        # In the first layer, please provide the input shape (32,32,3)
        model.add(
            layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))  # Result will be 32 30*30 outputs
        model.add(layers.MaxPooling2D((2, 2)))  # Result will be 32 15*15 outputs
        model.add(layers.Conv2D(64, (3, 3)))  # Result will be 64 13*13 outputs
        model.add(layers.MaxPooling2D((2, 2)))  # Result will be 64 6*6 outputs
        model.add(layers.Conv2D(64, (3, 3), activation='softmax'))  # Result will be 64 4*4 outputs

        model.add(layers.Flatten())  # Result will be a vector with length 4*4*64 = 1024
        if use_relu:
            model.add(layers.Dense(128, activation='relu'))  # Result will 64 outputs
            model.add(layers.Dense(64, activation='relu'))  # Result will 64 outputs
        else:
            model.add(layers.Dense(128, activation='sigmoid'))  # Result will 64 outputs
            model.add(layers.Dense(64, activation='sigmoid'))  # Result will 64 outputs

        model.add(layers.Dense(10, activation='softmax'))  # Result will be 10 outputs

        print(model.summary())
        
        if topK <= 1:
            model.compile(optimizer=optimizer_config, loss=loss_fn,
                      metrics=['accuracy'])
            
            training_history = model.fit(train_images, train_labels, epochs=epochs,
                                        validation_data=(val_images, val_labels))

            test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
            print("Final Accuracy achieved is: ", test_accuracy)

        else:
            model.compile(optimizer=optimizer_config, 
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=topK)])
            
            training_history = model.fit(train_images, train_labels, epochs=epochs,
                                        validation_data=(val_images, val_labels))

            test_loss, test_accuracy, test_topk_accuracy = model.evaluate(test_images, test_labels, verbose=2)
            print("Final Accuracy achieved is: ", test_accuracy, " with top-K accuracy as", test_topk_accuracy)
        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")

    else:
        print("Model found, there is no need to re-train the model ...")


def train_mnist_3_layer_mlp(train_data, test_data, path, overwrite=False, use_relu=True,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                            epochs=20):
    if use_relu:
        print(" >> ACTIVATION BY ReLU ...")
    else:
        print(" >> ACTIVATION BY Sigmoid ...")

    (x, y)=train_data
    (test_images, test_labels)=test_data

    train_images, val_images , train_labels, val_labels = train_test_split(x, y, test_size=0.167, train_size=0.833)

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print("Model not found, create and train a new one ...")
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=(28,28,1)))
        if not use_relu:
            model.add(layers.Dense(128, activation='sigmoid'))
        else:
            model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        print(model.summary())
        model.compile(optimizer=optimizer_config, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        training_history = model.fit(train_images, train_labels, epochs=epochs,
                                     validation_data=(val_images, val_labels))

        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
        print("Final Accuracy achieved is: ", test_accuracy)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")
        '''
        plt.plot(training_history.history['accuracy'], label="Accuracy")
        plt.plot(training_history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.8,1])
        plt.legend(loc='lower right')
        #plt.show()
        '''
    else:
        print("Model found, there is no need to re-train the model ...")


def train_mnist_cnn(train_data, test_data, path, overwrite=False, use_relu=True,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                            epochs=20):
    # Ref: https://www.kaggle.com/code/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1
    # Annotations: Experiment No. 3
    (x, y)=train_data
    (test_images, test_labels)=test_data

    #train_images, val_images , train_labels, val_labels = train_test_split(x, y, test_size=0.167, train_size=0.833)

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print("Model not found, create and train a new one ...")
        model = models.Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
        model.add(layers.Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))

        model.add(layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2))) 

        model.add(layers.Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
            
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        print(model.summary())
        model.compile(optimizer=optimizer_config, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        training_history = model.fit(x, y, epochs=epochs,
                                     validation_data=(test_images, test_labels))

        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
        print("Final Accuracy achieved is: ", test_accuracy)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")

    else:
        print("Model found, there is no need to re-train the model ...")


def train_mnist_5_layer_mlp(train_data, test_data, path, overwrite=False, use_relu=True,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                            epochs=20):
    if use_relu:
        print(" >> ACTIVATION BY ReLU ...")
    else:
        print(" >> ACTIVATION BY Sigmoid ...")

    (x, y)=train_data
    (test_images, test_labels)=test_data

    train_images, val_images , train_labels, val_labels = train_test_split(x, y, test_size=0.167, train_size=0.833)

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print("Model not found, create and train a new one ...")
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=(28,28,1)))
        if not use_relu:
            model.add(layers.Dense(128, activation='sigmoid'))
            model.add(layers.Dense(128, activation='sigmoid'))
            model.add(layers.Dense(64, activation='sigmoid'))
        else:
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        print(model.summary())
        model.compile(optimizer=optimizer_config, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        training_history = model.fit(train_images, train_labels, epochs=epochs,
                                     validation_data=(val_images, val_labels))

        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
        print("Final Accuracy achieved is: ", test_accuracy)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")
        '''
        plt.plot(training_history.history['accuracy'], label="Accuracy")
        plt.plot(training_history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.8,1])
        plt.legend(loc='lower right')
        #plt.show()
        '''
    else:
        print("Model found, there is no need to re-train the model ...")


def train_pneumonia_binary_classification_cnn(train_data,
                                              test_data,
                                              path,
                                              overwrite=False,
                                              epochs=20,
                                              val_data=None,
                                              data_augmentation=False,
                                              img_size=64):


    (test_images, test_labels)=test_data
    if val_data is None:
        (x, y) = train_data
        train_images, val_images , train_labels, val_labels = train_test_split(x, y, test_size=0.167, train_size=0.833)
    else:
        (train_images, train_labels) = train_data
        (val_images, val_labels) = val_data

    # With data augmentation to prevent overfitting and handling the imbalance in dataset
    if data_augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.2,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        datagen.fit(train_images)

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print("Model not found, create and train a new one ...")

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), strides=1, padding='same', activation='relu',
                                input_shape=(img_size, img_size, 1)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))

        model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))

        model.add(layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))

        model.add(layers.Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))

        model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2, 2), strides=2, padding='same'))

        model.add(layers.Flatten())
        model.add(layers.Dense(units=128, activation='relu'))
        model.add(layers.Dense(units=1, activation='sigmoid'))

        model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

        print(model.summary())

        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.3,
                                                    min_lr=0.000001)

        training_history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                            epochs=epochs,
                            validation_data=datagen.flow(val_images, val_labels),
                            callbacks=[learning_rate_reduction])


        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
        print("Final Accuracy achieved is: ", test_accuracy)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")
        '''
        plt.plot(training_history.history['accuracy'], label="Accuracy")
        plt.plot(training_history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.7,1])
        plt.legend(loc='lower right')
        #plt.show()
        '''
    else:
        print("Model found, there is no need to re-train the model ...")


def train_cifar_100_9_layer_cnn(train_data,
                            test_data,
                            valid_data,
                            path,
                            overwrite=False,
                            use_relu=False,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                            epochs=30):

    (train_images, train_labels)=train_data
    (test_images, test_labels)=test_data
    (valid_images, valid_labels)=valid_data

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print(path, " - model not found, create and train a new one ...")
        model = models.Sequential()
        # In the first layer, please provide the input shape (32,32,3)
        
        model.add(layers.Conv2D(input_shape=(32, 32, 3), kernel_size=(2, 2), padding='same', strides=(2, 2), filters=32))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
        model.add(layers.Conv2D(kernel_size=(2, 2), padding='same', strides=(2, 2), filters=64))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
        model.add(layers.Conv2D(64, (2, 2), activation='softmax'))  # Result will be 64 4*4 outputs
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
        model.add(layers.Flatten())

        if use_relu:
            model.add(layers.Dense(256, activation='relu'))  # Result will 256 outputs
            model.add(layers.Dense(128, activation='relu'))  # Result will 128 outputs
        else:
            model.add(layers.Dense(256, activation='sigmoid'))  # Result will 64 outputs
            model.add(layers.Dense(128, activation='sigmoid'))  # Result will 64 outputs

        model.add(layers.Dense(100, activation='softmax'))  # Result will be 10 outputs

        print(model.summary())
        
        model.compile(optimizer=optimizer_config, 
                    loss=loss_fn,
                    metrics=['sparse_top_k_categorical_accuracy'])
        
                      
        augmenter = ImageDataGenerator(rescale=1.0/255.,
                             rotation_range=20,
                             width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             shear_range=0.1, 
                             zoom_range=0.2, 
                             fill_mode='nearest',
                             horizontal_flip=True)
        rescalator=ImageDataGenerator(rescale=1.0/255.)

        train_generator=augmenter.flow(train_images, train_labels, batch_size=20)
        valid_generator=rescalator.flow(valid_images, valid_labels, batch_size=20)
        test_generator=rescalator.flow(test_images, test_labels, batch_size=20)


        batch_size=20
        steps_per_epoch=train_generator.n//batch_size
        validation_steps=valid_generator.n//batch_size

        training_history = model.fit(train_generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            validation_data=valid_generator,
                            validation_steps=validation_steps
                            )

        test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
        print("Final Accuracy achieved is: ", test_accuracy)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")
        
    else:
        print("Model found, there is no need to re-train the model ...")


# DEPRECATED
def train_cifar_6_layer_mlp(train_data, test_data, path, overwrite=False,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001)):
    (train_images, train_labels)=train_data
    (test_images, test_labels)=test_data

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print("Model not found, create and train a new one ...")
        model = models.Sequential()

        # In the first layer, please provide the input shape (32,32,3)
        model.add(layers.Flatten(input_shape=(32,32,3)))
        model.add(layers.Dense(1536, activation='relu'))
        model.add(layers.Dense(768, activation='relu'))
        model.add(layers.Dense(384, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(10)) # Result will be 10 outputs

        print(model.summary())
        model.compile(optimizer=optimizer_config, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        training_history = model.fit(train_images, train_labels, epochs=100,
                                     validation_data=(test_images, test_labels))

        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
        print("Final Accuracy achieved is: ", test_accuracy)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")
        '''
        plt.plot(training_history.history['accuracy'], label="Accuracy")
        plt.plot(training_history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.4, 1])
        plt.legend(loc='lower right')
        plt.show()
        '''
    else:
        print("Model found, there is no need to re-train the model ...")


def transfer_vgg_19_cifar(train_data, test_data, path, overwrite=False, 
                        optimizer_config = "RMSprop",
                        loss_fn ="categorical_crossentropy",
                        epochs=20):

    (train_images, train_labels)=train_data
    (test_images, test_labels)=test_data

    train_images = resize_img(train_images)
    test_images = resize_img(test_images)

    # Transform all labels to one-hot encoding
    train_labels = to_categorical(train_labels,num_classes=10)
    test_labels = to_categorical(test_labels,num_classes=10)

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print(path, " - model not found, create and train a new one ...")
        
        # Include top = add fully connected layers to layer.
        # Weights = use pretrained weights (trained in imagenet)
        vgg = VGG19(include_top=False,weights="imagenet",input_shape=(48,48,3))

        model = models.Sequential()
        for layer in vgg.layers:
            model.add(layer)
        
        # Ensure the vgg layers are not trainable
        for layer in model.layers:
            layer.trainable = False
        
        # Adding (trainable) fully connected layers
        model.add(layers.Flatten())
        model.add(layers.Dense(256))
        model.add(layers.Dense(128))
        model.add(layers.Dense(10,activation="softmax"))

        print(model.summary())

        # Total params: 20,091,338
        # Trainable params: 66,954
        # Non-trainable params: 20,024,384

        model.compile(optimizer=optimizer_config, loss=loss_fn, metrics=["accuracy"])
        hist = model.fit(train_images,train_labels,validation_split=0.15,epochs=epochs,batch_size=1000)


        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
        print("Final Accuracy achieved is:", test_accuracy, "and the loss is:", test_loss)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")
    else:
        print("Model found, there is no need to re-train the model ...")

'''
Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
Connect the feature extraction and "classifier" layers to build the model.
'''
def transfer_resnet_50(train_data, test_data, path, overwrite=False, 
                        optimizer_config = "SGD",
                        loss_fn ="sparse_categorical_crossentropy",
                        epochs=3):

    (train_images, train_labels)=train_data
    (test_images, test_labels)=test_data

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print(path, " - model not found, create and train a new one ...")
        
        inputs = tf.keras.layers.Input(shape=(32,32,3))
    
        resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)

        resnet_feature_extractor = transfer_resnet_feature_extractor(resize)
        classification_output = transfer_resnet_entail_classifier(resnet_feature_extractor)
        model = tf.keras.Model(inputs=inputs, outputs = classification_output)
        print(model.summary())

        # Total params: 26,215,818
        # Trainable params: 26,162,698
        # Non-trainable params: 53,120
        
        model.compile(optimizer=optimizer_config, 
                        loss=loss_fn,
                        metrics = ['accuracy'])

        hist = model.fit(train_images, train_labels, epochs=epochs, 
                            validation_data = (test_images, test_labels), batch_size=64)
        test_loss, test_accuracy = model.evaluate(test_images, test_labels, batch_size=64)
        print("Final Accuracy achieved is:", test_accuracy, "and the loss is:", test_loss)

        model.save(path)
        print("Model has been saved")
    else:
        print("Model found, there is no need to re-train the model ...")
  

'''
Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
Input size is 224 x 224.
'''
def transfer_resnet_feature_extractor(inputs):

  feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')(inputs)
  return feature_extractor


'''
Defines final dense layers and subsequent softmax layer for classification.
'''
def transfer_resnet_entail_classifier(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
    return x


def resize_img(img):
    num_imgs = img.shape[0]
    new_array = np.zeros((num_imgs, 48,48,3))
    for i in range(num_imgs):
        new_array[i] = cv2.resize(img[i,:,:,:],(48,48))
    return new_array

def preprocess_image_input_resnet(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims