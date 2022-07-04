#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

# Import publicly published & installed packages
import tensorflow as tf
from numpy.random import seed
import os, time, csv, shutil, math, time
from tensorflow.python.eager.monitoring import Sampler

# Import in-house classes
from paoding.sampler import Sampler
from paoding.evaluator import Evaluator
from paoding.utility.option import SamplingMode, ModelType
import paoding.utility.utils as utils
import paoding.utility.bcolors as bcolors
import paoding.utility.simulated_propagation as simprop

class Pruner:

    constant = 0
    model = None
    optimizer = None
    sampler = None
    robustness_evaluator = None
    model_path = None
    test_set = None

    pruning_target = None
    pruning_step = None
    
    model_type = -1

    lo_bound = 0
    hi_bound = 1
    
    def __init__(self, path, test_set=None, target=0.5, step=0.025, sample_strategy=None, input_interval=(0,1), model_type=ModelType.XRAY, seed_val=None):
        """
        Initializes `Pruner` class.
        Args:     
        path: The path of neural network model to be pruned.
        test_set: The tuple of test features and labels used for evaluation purpose.
        target: The percentage value of expected pruning goal (optional, 0.50 by default).
        step: The percentage value of pruning portion during each epoch (optional, 0.025 by default).
        sample_strategy: The sampling strategy specified for pruning (optional).
        alpha: The value of alpha parameters to be used in stochastic mode (optional, 0.75 by default).
        input_interval: The value range of an legal input (optional, [0,1] by default).
        model_type: The enumerated value that specifies the model type (optional, binary classification model by default).
            [PS] 4 modes are supported in the Alpha release, refer to the ``paoding.utility.option.ModelType`` for the technical definition.
        seed: The seed for randomization for the reproducibility purpose (optional, to use only for the experimental purpose)
        """
        if sample_strategy == None:
            self.sampler = Sampler()
        else:
            self.sampler = sample_strategy
        self.robustness_evaluator = Evaluator()
        
        self.model_path = path
        # Specify a random seed
        if seed_val is not None:
            seed(seed_val)
            tf.random.set_seed(seed_val)

        self.model_type = model_type

        self.target_adv_epsilons = [0.5]

        self.pruning_target = target
        self.pruning_step = step
        
        self.evaluation_batch = 50

        # E.g. EPOCHS_PER_CHECKPOINT = 5 means we save the pruned model as a checkpoint after each five
        #    epochs and at the end of pruning
        self.EPOCHS_PER_CHECKPOINT = 15
        
        self.test_set = test_set

        (self.lo_bound, self.hi_bound) = input_interval
        #self.first_mlp_layer_size = first_mlp_layer_size

    def load_model(self, optimizer=None):
        """
        Load the model.
        Args: 
        optimizer: The optimizer specified for evaluation purpose (optional, RMSprop with lr=0.01 by default).
        """
        self.model = tf.keras.models.load_model(self.model_path)
        print(self.model.summary())
        
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        else:
            self.optimizer = optimizer

    def save_model(self, path):
        """
        Save the model to the path specified.
        Args: 
        path: The path that the model to be saved.
        """
        if os.path.exists(path):
            shutil.rmtree(path)
            print("Overwriting existing pruned model ...")

        self.model.save(path)
        print(" >>> Pruned model saved")
       
    def evaluate(self, metrics=['accuracy']):
        """
        Evaluate the model performance.
        Args: 
        metrics: The list of TF compatible metrics (optional, accuracy (only) by default).
        Returns:
        A tuple of loss and accuracy values
        """
        if self.test_set is None:
            print("Test set not provided, evaluation aborted...")
            return 0, 0

        test_features, test_labels = self.test_set
        # self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=metrics)
        loss, accuracy = self.model.evaluate(test_features, test_labels, verbose=2)
        print("Evaluation accomplished -- [ACC]", accuracy, "[LOSS]", loss)   
        return loss, accuracy
       
    def prune(self, evaluator=None, pruned_model_path=None):
        """
        Perform pruning and save the pruned model to a specified location.
        Args: 
        evaluator: The evaluation configuration (optional, no evaluation requested by default).
        pruned_model_path: The location to save the pruned model (optional, a fixed path by default).
        """
        if evaluator is not None:
            self.robustness_evaluator = evaluator
            self.target_adv_epsilons = evaluator.epsilons
            self.evaluation_batch = evaluator.batch_size
        test_images, test_labels = self.test_set
        utils.create_dir_if_not_exist("paoding/logs/")
        # utils.create_dir_if_not_exist("paoding/save_figs/")
        
        if pruned_model_path is None:
            pruned_model_path=self.model_path+"_pruned"

        # Define a list to record each pruning decision
        tape_of_moves = []
        # Define a list to record benchmark & evaluation per pruning epoch (begins with original model)
        score_board = []
        accuracy_board = []

        ################################################################
        # Launch a pruning epoch                                       #
        ################################################################

        epoch_couter = 0
        num_units_pruned = 0
        percentage_been_pruned = 0
        stop_condition = False
        neurons_manipulated =None
        target_scores = None
        pruned_pairs = None
        cumulative_impact_intervals = None
        saliency_matrix=None
        
        model = self.model

        big_map = simprop.get_definition_map(model, input_interval=(self.lo_bound, self.hi_bound))
    
        # Start elapsed time counting
        start_time = time.time()

        while(not stop_condition):

            pruning_result_dict = self.sampler.nominate(model,big_map, 
                                                prune_percentage=self.pruning_step,
                                                cumulative_impact_intervals=cumulative_impact_intervals,
                                                neurons_manipulated=neurons_manipulated, saliency_matrix=saliency_matrix,
                                                bias_aware=True)

            model = pruning_result_dict['model']
            neurons_manipulated = pruning_result_dict['neurons_manipulated']
            target_scores = pruning_result_dict['target_scores']
            pruned_pairs = pruning_result_dict['pruned_pairs']
            cumulative_impact_intervals = pruning_result_dict['cumulative_impact_intervals']
            saliency_matrix = pruning_result_dict['saliency_matrix']
            score_dicts = pruning_result_dict['pruning_pairs_dict_overall_scores']

            epoch_couter += 1

            # Check if the list of pruned pair is empty or not - empty means no more pruning is feasible
            num_pruned_curr_batch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        num_pruned_curr_batch += len(pairs)

            if num_pruned_curr_batch == 0:
                stop_condition = True
                print(" >> No more hidden unit could be pruned, we stop at EPOCH", epoch_couter)
            else:
                if not self.sampler.mode == SamplingMode.BASELINE:
                    print(" >> Cumulative impact as intervals after this epoch:")
                    print(cumulative_impact_intervals)

                percentage_been_pruned += self.pruning_step
                print(" >> Pruning progress:", bcolors.BOLD, str(round(percentage_been_pruned * 100, 2)) + "%", bcolors.ENDC)

                model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
                if evaluator is not None and self.test_set is not None:                    
                    robust_preservation = self.robustness_evaluator.evaluate_robustness(model, (test_images, test_labels), self.model_type)
                    #loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
                    loss, accuracy = self.evaluate()

                    # Update score_board and tape_of_moves
                    score_board.append(robust_preservation)
                    accuracy_board.append((round(loss, 4), round(accuracy, 4)))
                    print(bcolors.OKGREEN + "[Epoch " + str(epoch_couter) + "]" + str(robust_preservation) + bcolors.ENDC)

                tape_of_moves.append(pruned_pairs)
                pruned_pairs = None
            # Check if have pruned enough number of hidden units
            if self.sampler.mode == SamplingMode.BASELINE and percentage_been_pruned >= 0.5:
                print(" >> Maximum pruning percentage has been reached")
                stop_condition = True
            elif not stop_condition and percentage_been_pruned >= self.pruning_target:
                print(" >> Target pruning percentage has been reached")
                stop_condition = True

            # Save the pruned model at each checkpoint or after the last pruning epoch
            if epoch_couter % self.EPOCHS_PER_CHECKPOINT == 0 or stop_condition:
                curr_pruned_model_path = pruned_model_path + "_ckpt_" + str(math.ceil(epoch_couter/self.EPOCHS_PER_CHECKPOINT))

                if os.path.exists(curr_pruned_model_path):
                    shutil.rmtree(curr_pruned_model_path)
                print("Overwriting existing pruned model ...")

                model.save(curr_pruned_model_path)
                print(" >>> Pruned model saved")

        # Stop elapsed time counting
        end_time = time.time()
        print("Elapsed time: ", round((end_time - start_time)/60.0, 3), "minutes /", int(end_time - start_time), "seconds")

        ################################################################
        # Save the tape of moves                                       #
        ################################################################
        
        # Obtain a timestamp
        local_time = time.localtime()
        timestamp = time.strftime('%b-%d-%H%M', local_time)


        tape_filename = "paoding/logs/chest-" + timestamp + "-" + str(self.evaluation_batch)
        if evaluator is None:
            tape_filename = tape_filename+"-BENCHMARK"

        if self.sampler.mode == SamplingMode.BASELINE:
            tape_filename += "_tape_baseline.csv"
        else:
            tape_filename = tape_filename + "_tape_" + self.sampler.mode.name + ".csv"

        if os.path.exists(tape_filename):
            os.remove(tape_filename)

        with open(tape_filename, 'w+', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_line = [str(eps) for eps in self.target_adv_epsilons]
            csv_line.append('moves,loss,accuracy')
            csv_writer.writerow(csv_line)

            for index, item in enumerate(score_board):
                rob_pres_stat = [item[k] for k in self.target_adv_epsilons]
                rob_pres_stat.append(tape_of_moves[index])
                rob_pres_stat.append(accuracy_board[index])
                csv_writer.writerow(rob_pres_stat)
            
            if evaluator is None:
                csv_writer.writerow(["Elapsed time: ", round((end_time - start_time) / 60.0, 3), "minutes /", int(end_time - start_time), "seconds"])

        print("Pruning accomplished")
 