import math

import tensorflow as tf

from src.model.modelc import build_modelc
from src.model.lenet import build_lenet5
from src.model.resnet import resnet_v2, resnet_v1
from src.model.stacked_lstm import build_stacked_lstm
from src.subspace.builder.model_builders import build_model_mnist_fc, \
    build_cnn_model_mnist_bhagoji, build_cnn_model_mnist_dev_conv, build_cnn_model_mnistcnn_conv, build_LeNet_cifar, \
    build_cnn_model_cifar_allcnn, build_model_cifar_LeNet_fastfood
from src.subspace.builder.resnet import build_LeNet_resnet, build_resnet_fastfood
from tensorflow.keras.regularizers import l2

from src.model.mobilenet import mobilenetv2_cifar10


class Model:
    @staticmethod
    def create_model(model_name, intrinsic_dimension=None, regularization_rate=None, disable_bn=False):
        """Creates NN architecture based on a given model name

        Args:
            model_name (str): name of a model
        """
        if model_name == 'mnist_cnn':
            do_fact = 0.3
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu',
                                       input_shape=(28, 28, 1), dtype=float),
                tf.keras.layers.MaxPooling2D(pool_size=2),
                # tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=2),
                # tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                #tf.keras.layers.Dense(32, activation='relu'),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
        elif model_name == 'vgg-cifar':
            regularizer = l2(regularization_rate) if regularization_rate is not None else None
            model = tf.keras.Sequential()

            model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                            input_shape=(32,32,3), kernel_initializer='he_normal', 
                            kernel_regularizer=regularizer, name='block1_conv1'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=regularizer, 
                            name='block1_conv2'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                            strides=(2,2), name='block1_pool'))

            model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                            kernel_initializer='he_normal', 
                            kernel_regularizer=regularizer, name='block2_conv1'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=regularizer, 
                            name='block2_conv2'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                            strides=(2,2), name='block2_pool'))
            
            model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                            kernel_initializer='he_normal', 
                            kernel_regularizer=regularizer, name='block3_conv1'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=regularizer, 
                            name='block3_conv2'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Conv2D(256, (3, 3), padding='same',activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=regularizer, 
                            name='block3_conv3'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                            strides=(2,2), name='block3_pool'))

            model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                            kernel_initializer='he_normal', 
                            kernel_regularizer=regularizer, name='block4_conv1'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=regularizer, 
                            name='block4_conv2'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=regularizer, 
                            name='block4_conv3'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                            strides=(2,2), name='block4_pool'))

            model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                            kernel_initializer='he_normal', 
                            kernel_regularizer=regularizer, name='block5_conv1'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=regularizer, 
                            name='block5_conv2'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same',activation='relu',
                            kernel_initializer='he_normal', kernel_regularizer=regularizer, 
                            name='block5_conv3'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                            strides=(2,2), name='block5_pool'))

            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(256, activation='relu', name='fc1',
                            kernel_initializer='he_normal', kernel_regularizer=regularizer))
            model.add(tf.keras.layers.Dense(256, activation='relu', name='fc2',
                            kernel_initializer='he_normal', kernel_regularizer=regularizer))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(10, activation='softmax', name='predictions',
                            kernel_initializer='he_normal'))
            model.summary()
            
        elif model_name == 'dev':
            regularizer = l2(regularization_rate) if regularization_rate is not None else None

            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=regularizer, bias_regularizer=regularizer),
                tf.keras.layers.Conv2D(4, (3, 3), activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizer, bias_regularizer=regularizer),
                tf.keras.layers.Dense(10, activation='softmax'),
            ])

        elif model_name == 'fc_mnist': 
            # LeCun et al. 1998
            model = tf.keras.Sequential([ # the model used in Paoding
                tf.keras.layers.Flatten(input_shape=(28,28,1)),
                tf.keras.layers.Dense(500, activation='relu'),
                tf.keras.layers.Dense(150, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax'),
            ])
        elif model_name == 'alexnet_cifar':
            model=tf.keras.models.Sequential([
                tf.keras.layers.UpSampling2D(size=(2,2), input_shape=(32,32,3)),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(64,64,3)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3,3)),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024,activation='relu'),
                tf.keras.layers.Dense(1024,activation='relu'),
                tf.keras.layers.Dense(10,activation='softmax')  
            ])
        elif model_name == 'bhagoji':
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, kernel_size=(5, 5), padding='valid', activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
                # tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
        elif model_name == 'lenet5_cifar':
            model = build_lenet5(input_shape=(32, 32, 3), l2_reg=regularization_rate)
        elif model_name == 'lenet5_mnist':
            model = build_lenet5(input_shape=(28, 28, 1), l2_reg=regularization_rate)
            model.summary()
        elif model_name == 'allcnn':
            model = build_modelc(l2_reg=regularization_rate)
            model.summary()
        elif model_name == 'allcnn_intrinsic':
            model = build_cnn_model_cifar_allcnn(vsize=intrinsic_dimension, weight_decay=regularization_rate)
        elif model_name == 'resnet18' or model_name == 'resnet20':
            model = resnet_v1(input_shape=(32, 32, 3), depth=20)
            model.summary()
        elif model_name == 'resnet18_mnist' or model_name == 'resnet20_mnist':
            model = resnet_v1(input_shape=(28, 28, 1), depth=20)
            model.summary()
        elif model_name == 'resnet32':
            model = resnet_v1(input_shape=(32, 32, 3), depth=32)
            model.summary()
        elif model_name == 'resnet44':
            model = resnet_v1(input_shape=(32, 32, 3), depth=44)
        elif model_name == 'resnet56':
            model = resnet_v1(input_shape=(32, 32, 3), depth=56)
            model.summary()
        elif model_name == 'resnet110':
            model = resnet_v1(input_shape=(32, 32, 3), depth=110)
        elif model_name == 'resnet18_v2' or model_name == 'resnet20_v2':
            model = resnet_v2(input_shape=(32, 32, 3), depth=20)
        elif model_name == 'resnet56_v2':
            model = resnet_v2(input_shape=(32, 32, 3), depth=56)
            model.summary()
            print("HI")
        elif model_name == 'resnet50':
            inputs = tf.keras.layers.Input(shape=(32,32,3))
            resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
            feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                include_top=False, weights='imagenet')(resize)
            x = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            x = tf.keras.layers.Dense(64, activation="relu")(x)
            x = tf.keras.layers.Dense(10, activation="softmax", name="classification")(x)
            model = tf.keras.Model(inputs=inputs, outputs = x)
        elif model_name == 'mobilenet':
            model = mobilenetv2_cifar10()
            model.summary()
        elif model_name == 'dev_fc_intrinsic':
            model, _ = build_model_mnist_fc(vsize=intrinsic_dimension, width=100)
        elif model_name == 'bhagoji_intrinsic':
            model = build_cnn_model_mnist_bhagoji(vsize=intrinsic_dimension, proj_type='sparse')
        elif model_name == 'dev_intrinsic':
            # model = build_model_cifar_LeNet_fastfood(vsize=intrinsic_dimension)
            model = build_cnn_model_mnist_dev_conv(vsize=intrinsic_dimension, proj_type='sparse', weight_decay=regularization_rate)
            Model.normalize(model)
        elif model_name == 'mnistcnn_intrinsic':
            model = build_cnn_model_mnistcnn_conv(vsize=intrinsic_dimension, proj_type='sparse')
        elif model_name =='lenet5_intrinsic':
            # model = build_lenet_cifar_old(intrinsic_dimension)
            model = build_LeNet_cifar(vsize=intrinsic_dimension, proj_type='sparse', weight_decay=0.001)
            Model.normalize(model)
        elif model_name =='resnet18_intrinsic':
            # model = build_lenet_cifar_old(intrinsic_dimension)
            model = build_LeNet_resnet(20, vsize=intrinsic_dimension, proj_type='sparse', weight_decay=0.001,
                                       disable_bn=disable_bn)
            # model = build_resnet_fastfood(20, vsize=intrinsic_dimension, proj_type='sparse', weight_decay=0.001)
            Model.normalize(model)
            model.summary()
        elif model_name == 'stacked_lstm':
            model = build_stacked_lstm()
            model.summary()
            return model
        else:
            raise Exception('model `%s` not supported' % model_name)

        return model

    @staticmethod
    def normalize(model, proj_type='sparse'):
        basis_matrices = []
        normalizers = []

        for layer in model.layers:
            try:
                basis_matrices.extend(layer.offset_creator.basis_matrices)
            except AttributeError:
                continue
            try:
                normalizers.extend(layer.offset_creator.basis_matrix_normalizers)
            except AttributeError:
                continue

        if proj_type == 'sparse':

            # Norm of overall basis matrix rows (num elements in each sum == total parameters in model)
            # bm_row_norms = tf.sqrt(tf.add_n([tf.sparse_reduce_sum(tf.square(bm), 1) for bm in basis_matrices]))
            # # Assign `normalizer` Variable to these row norms to achieve normalization of the basis matrix
            # # in the TF computational graph
            # rescale_basis_matrices = [tf.assign(var, tf.reshape(bm_row_norms, var.shape)) for var in normalizers]
            # _ = sess.run(rescale_basis_matrices)
            bm_row_norms = tf.sqrt(tf.add_n([tf.sparse.reduce_sum(tf.square(bm), 1) for bm in basis_matrices]))
            for var in normalizers:
                var.assign(tf.reshape(bm_row_norms, var.shape))

        elif proj_type == 'dense':
            bm_sums = [tf.reduce_sum(tf.square(bm), 1) for bm in basis_matrices]
            divisor = tf.expand_dims(tf.sqrt(tf.add_n(bm_sums)), 1)
            rescale_basis_matrices = [tf.assign(var, var / divisor) for var in basis_matrices]
            _ = sess.run(rescale_basis_matrices)

    @staticmethod
    def model_supported(model_name, dataset_name):
        supported_types = {
            "mnist": ["mnist_cnn", "dev", "bhagoji", "dev_fc_intrinsic", "dev_intrinsic", "mnistcnn_intrinsic", "bhagoji_intrinsic", "lenet5_mnist", "resnet18_mnist"],
            "fmnist": ["mnist_cnn", "dev", "bhagoji", "dev_fc_intrinsic", "dev_intrinsic", "mnistcnn_intrinsic", "bhagoji_intrinsic", "lenet5_mnist", "resnet18_mnist"],
            "femnist": ["mnist_cnn", "dev", "bhagoji", "dev_fc_intrinsic", "dev_intrinsic", "mnistcnn_intrinsic", "bhagoji_intrinsic", "lenet5_mnist", "resnet18_mnist"],
            "cifar10": ["resnet18", "resnet32", "resnet44", "resnet50", "resnet56", "resnet110", "resnet18_v2", "resnet56_v2", "lenet5_cifar", "lenet5_intrinsic", "allcnn", "allcnn_intrinsic"]
        }
        return model_name in supported_types[dataset_name]

    @staticmethod
    def model_supports_weight_analysis(model_name):
        return model_name not in ["dev_intrinsic", "dev_fc_intrinsic", "bhagoji_intrinsic", 
            "mnistcnn_intrinsic", "allcnn", "allcnn_intrinsic",
            "resnet50"]

    @staticmethod
    def create_optimizer(optimizer_name, learning_rate, decay, steps_per_round):
        """Creates optimizer based on given parameters

        Args:
            optimizer_name (str): name of the optimizer
            learning_rate (float|object): initial learning rate
            decay (src.config.definitions.LearningDecay|None): type of decay
            steps_per_round (int): number of optimizer steps per round

        Returns:
            keras optimizer
        """
        if decay is not None:
            lr_schedule = Model.current_lr(learning_rate, decay.type,
                                           decay.decay_steps, decay.decay_rate, decay.decay_boundaries, decay.decay_values,
                                           decay.step_epochs, steps_per_round)
        else:
            lr_schedule = learning_rate
        if optimizer_name == 'Adam':
            return tf.keras.optimizers.Adam(lr_schedule)
        elif optimizer_name == 'SGD':
            return tf.keras.optimizers.SGD(lr_schedule, 0.9)

        raise Exception('Optimizer `%s` not supported.' % optimizer_name)

    @staticmethod
    def current_lr(learning_rate, decay_type, decay_steps, decay_rate, decay_boundaries, decay_values, steps_epoch, steps_per_batch):
        # lr = learning_rate * \
        #      math.pow(decay_rate, math.floor(epoch / decay_steps))

        # lr = learning_rate * \
        #     tf.pow(decay_rate, tf.cast(tf.floor(epoch / decay_steps), dtype=tf.float32))

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     learning_rate,
        #     decay_steps=decay_steps,
        #     decay_rate=decay_rate,
        #     staircase=False)
        steps_multiplier = 1
        if steps_epoch:
            steps_multiplier = steps_per_batch

        if decay_type == 'exponential':
            # exp
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate,
                decay_steps=decay_steps * steps_multiplier,
                decay_rate=decay_rate,
                staircase=False)
            return lr_schedule
        elif decay_type == 'boundaries':
            values = [learning_rate * v for v in decay_values]
            boundaries = [boundary * steps_multiplier for boundary in decay_boundaries]
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries, values)
            return lr_schedule
        else:
            return learning_rate

        # if epoch > 300 * 2:
        #     learning_rate *= 1e-1
        # if epoch > 250 * 2:
        #     learning_rate *= 1e-1
        # if epoch > 200 * 2:
        #     learning_rate *= 1e-1
        # print('Learning rate: ', lr)
        # return lr_schedule