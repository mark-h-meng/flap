#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

# Import publicly published & installed packages
import tensorflow as tf

# Import in-house classes
from paoding.utility.option import ModelType, AttackAlogirithm
import paoding.utility.adversarial_mnist_fgsm_batch as adversarial

class Evaluator:

    epsilons = []
    batch_size = 0
    attack_mode = -1
    metrics = ['accuracy']

    def __init__(self, epsilons = [0.5], batch_size = 50, attack_mode = AttackAlogirithm.FGSM, k=1):
        """Initializes `Evaluator` class.
        Args:
        epsilons: The collection of adversarial epsilons (optional, 0.5 only by default).
            Please refer to the FGSM for more details of the epsilon parameter.
        batch_size: The batch size of test samples for each pruning epoch (optional, 50 by default).
        attack_mode: The enumerated value to specify the attack algorithm applied for robustness preservation evaluation (optional, FGSM by default).
            [PS] Only FGSM is supported in the Alpha release, refer to the ``paoding.utility.option.AttackAlogirithm`` for the technical definition.
        k: The value indicates if the top-k accuracy is used (optiona, 1 by default).  
        """
        if type(epsilons) == list:
            self.epsilons = epsilons
        else:
            self.epsilons = [epsilons]
        self.batch_size = batch_size
        self.attack_mode = attack_mode

        if k > 1:
            self.metrics.append(tf.keras.metrics.TopKCategoricalAccuracy(k))

    def get_epsilons(self): 
        """Retrieve the epsilon parameters.
        Returns:
        epsilons: The collection of adversarial epsilons.
        """
        return self.epsilons

    def get_batch_size(self): 
        """Retrieve the batch size parameters.
        Returns:
        batch_size: The batch size of test samples for each pruning epoch.
        """
        return self.batch_size
    
    def set_epsilons(self, epsilons): 
        """Set the epsilon parameters.
        Args:
        epsilons: The collection of adversarial epsilons.
        """
        self.epsilons = epsilons

    def set_batch_size(self, batch_size): 
        """Set the batch size parameters.
        Args:
        batch_size: The batch size of test samples for each pruning epoch.
        """
        self.batch_size = batch_size

    def evaluate_robustness(self, model, test_set, model_type, k=1):
        """
        Evaluate the model performance.
        Args: 
        model: The neural network model to be used for robustness preservation evaluation.
        test_set: The tuple of test features and labels to be used for the evaluation.
        model_type: The enumerated value that specifies the model type.
        k:  The value indicates if the top-k accuracy is used (optiona, 1 by default).
        Returns:
        A dictionary of evaluation outcome, with each key represents the epsilon value, and value provides the number of robust instances observed.
        """
        if self.attack_mode == AttackAlogirithm.FGSM:
            return self.__fgsm(model, test_set, model_type, k)
        else:
            print("Evaluation mode not set or set to an illegal value, please check!")
        
    def __fgsm(self, model, test_set, model_type, k):
        test_features, test_labels = test_set
        if model_type == ModelType.XRAY:
            robust_preservation = adversarial.robustness_evaluation_chest(model,
                                                                (test_features, test_labels),
                                                                self.epsilons,
                                                                self.batch_size)
        elif model_type == ModelType.CREDIT:
            robust_preservation = adversarial.robustness_evaluation_kaggle(model,
                                                                (test_features, test_labels),
                                                                self.epsilons,
                                                                self.batch_size)
        elif model_type == ModelType.MNIST:
            robust_preservation = adversarial.robustness_evaluation(model,
                                                                (test_features, test_labels),
                                                                self.epsilons,
                                                                self.batch_size)
        elif model_type == ModelType.CIFAR:
            if k > 1:
                robust_preservation = adversarial.robustness_evaluation_cifar_topK(model, 
                                                                (test_features, test_labels),
                                                                self.epsilons,
                                                                self.batch_size, k)

            else:
                robust_preservation = adversarial.robustness_evaluation_cifar(model,
                                                                (test_features, test_labels),
                                                                self.epsilons,
                                                                self.batch_size)
        else:
            print("Robustness evaluation not available for this release!")
        return robust_preservation

