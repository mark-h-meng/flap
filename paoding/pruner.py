#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

# Import publicly published & installed packages
import tensorflow as tf
from numpy.random import seed
import os, time, csv, shutil, math, time
from pathlib import Path
from datetime import datetime
import numpy as np

# Import in-house classes
from paoding.sampler import Sampler
from paoding.evaluator import Evaluator
from paoding.utility.option import SamplingMode, ModelType
import paoding.utility.utils as utils
import paoding.utility.bcolors as bcolors
import paoding.utility.simulated_propagation as simprop
#import paoding.utility.model_profiler.profiler as profiler
import paoding.utility.dense_layer_surgeon as surgeon

class Pruner:

    constant = 0
    model = None
    optimizer = None
    loss = None
    sampler = None
    robustness_evaluator = None
    model_path = None
    test_set = None

    pruning_target = None
    pruning_step = None
    
    surgery_mode = False

    lo_bound = 0
    hi_bound = 1

    stepwise_cnn_pruning = False
    
    def __init__(self, path, test_set=None, target=0.5, step=0.025, sample_strategy=None, input_interval=(0,1), model_type=ModelType.XRAY, seed_val=None, stepwise_cnn_pruning=False, surgery_mode=False):
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

        self.target_adv_epsilons = [0.01, 0.05, 0.1]

        self.pruning_target = target
        self.pruning_step = step
        
        self.evaluation_batch = 50

        # E.g. EPOCHS_PER_CHECKPOINT = 5 means we save the pruned model as a checkpoint after each five
        #    epochs and at the end of pruning
        self.EPOCHS_PER_CHECKPOINT = 1000
        
        self.test_set = test_set

        (self.lo_bound, self.hi_bound) = input_interval
        #self.first_mlp_layer_size = first_mlp_layer_size

        self.stepwise_cnn_pruning = stepwise_cnn_pruning

        self.surgery_mode = surgery_mode

    def load_model(self, optimizer=None, loss=None):
        """
        Load the model.
        Args: 
        optimizer: The optimizer specified for evaluation purpose (optional, RMSprop with lr=0.01 by default).
        """
        self.model = tf.keras.models.load_model(self.model_path)
        print(self.model.summary())
        
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        else:
            self.optimizer = optimizer

        if loss is None:
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            self.loss = loss

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
       
    
    def evaluate(self, verbose=0, batch_size=None):
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
        
        # In case some test set is not simply a tuple, but a tensorflow DirectoryIterator, we need to directly
        ## feed the test_set into the evaluation function
        if type(self.test_set) is tuple:
            t_features, t_labels = self.test_set
            startTime = datetime.now()
            loss, accuracy = self.model.evaluate(t_features, t_labels, verbose=2, batch_size=batch_size)
            elapsed = datetime.now() - startTime
        else:
            startTime = datetime.now()
            loss, accuracy = self.model.evaluate(self.test_set, verbose=2, batch_size=batch_size)
            elapsed = datetime.now() - startTime
        if verbose > 0:
            print("Evaluation accomplished -- [ACC]", accuracy, "[LOSS]", loss, "[Elapsed Time]", elapsed)   
        return loss, accuracy

    #def profile(self):
    #    print(profiler.model_profiler(self.model, batch_size=1))


    def quantization(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        tflite_models_dir = Path('paoding/models/tflite_models/')
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        model_filename = self.model_type.name + ".tflite" 
        tflite_model_file = tflite_models_dir/model_filename  
        tflite_model_file.write_bytes(tflite_model)
        print(" >> Size after pruning:", os.path.getsize(tflite_model_file))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16_model = converter.convert()
        model_filename_f16 = self.model_type.name + "_quant_f16.tflite"
        tflite_model_fp16_file = tflite_models_dir/model_filename_f16
        tflite_model_fp16_file.write_bytes(tflite_fp16_model)
        print(" >> Size after quantization:", os.path.getsize(tflite_model_fp16_file))

    def prune(self, evaluator=None, save_file=False, pruned_model_path=None, verbose=0, model_name=None):
        """
        Perform fully connected pruning and save the pruned model to a specified location.
        Args: 
        evaluator: The evaluation configuration (optional, no evaluation requested by default).
        pruned_model_path: The location to save the pruned model (optional, a fixed path by default).
        """

        if not self.stepwise_cnn_pruning:
            " >> Stepwise CNN pruning enabled: CNN pruning will be done together with FC pruning per step"
            self.prune_cnv(evaluator, save_file, pruned_model_path, verbose)
        self.prune_fc(evaluator, save_file, pruned_model_path, verbose, model_name, include_cnn_per_step=self.stepwise_cnn_pruning)

    def prune_fc(self, evaluator=None, save_file=False, pruned_model_path=None, verbose=0, model_name=None, include_cnn_per_step=False):
        no_fc_to_prune = False
        progress_if_no_fc_to_prune = 0

        if evaluator is not None:
            self.robustness_evaluator = evaluator
            self.target_adv_epsilons = evaluator.epsilons
            self.evaluation_batch = evaluator.batch_size

        if model_name is None:
            model_name = self.model_type.name
            
        utils.create_dir_if_not_exist("paoding/logs/")
        # utils.create_dir_if_not_exist("paoding/save_figs/")
        
        if save_file and pruned_model_path is None:
            #pruned_model_path=self.model_path+"_pruned"
            pruned_model_path=self.model_path

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
        pruned_pairs_all_steps = None
        cumulative_impact_intervals = None
        saliency_matrix=None
        
        map_defined = False

        model = self.model
        
        # Start elapsed time counting
        start_time = time.time()

        while(not stop_condition):
            
            if include_cnn_per_step:
                pruned_model_path_conv = self.model_path + "conv_pruned"
                self.model = self.prune_cnv_step(None, save_file=True, pruned_model_path=pruned_model_path_conv, verbose=1)
                self.model = tf.keras.models.load_model(pruned_model_path_conv)
                self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
                print(self.model.summary())
                model = self.model

            try:
                if not map_defined:
                    big_map = simprop.get_definition_map(model, input_interval=(self.lo_bound, self.hi_bound))
                    map_defined = True
            except Exception as err:
                no_fc_to_prune = True
                print("Unexpected "+str(err))

            if no_fc_to_prune:
                
                loss, accuracy = self.evaluate(verbose=1)
                accuracy_board.append((round(loss, 4), round(accuracy, 4)))
                tape_of_moves.append([])

                progress_if_no_fc_to_prune += self.pruning_step
                if self.pruning_target <= progress_if_no_fc_to_prune:
                    stop_condition = True

                # Skip the current round of fc pruning
                continue


            pruned_pairs = None
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

            self.model = model
            epoch_couter += 1

            # Check if the list of pruned pair is empty or not - empty means no more pruning is feasible
            num_pruned_curr_batch = 0

            if pruned_pairs_all_steps is None and pruned_pairs is not None:
                pruned_pairs_all_steps = [[] for i in range(len(pruned_pairs))]
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        num_pruned_curr_batch += len(pairs)
                        for pair in pairs:
                            pruned_pairs_all_steps[layer].append(pair)

            if num_pruned_curr_batch == 0:
                stop_condition = True
                if verbose > 0:
                    print(" [DEBUG] No more hidden unit could be pruned, we stop at EPOCH", epoch_couter)
            else:
                if not self.sampler.mode == SamplingMode.BASELINE:
                    if verbose > 0:
                        print(" [DEBUG] Cumulative impact as intervals after this epoch:")
                        print(cumulative_impact_intervals)

                percentage_been_pruned += self.pruning_step
                print(" >> Pruned", num_pruned_curr_batch, "hidden units in this epoch")
                print(" >> Pruning progress:", bcolors.BOLD, str(round(percentage_been_pruned * 100, 2)) + "%", bcolors.ENDC)

                self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
                
                if evaluator is not None and self.test_set is not None:                    
                    robust_preservation = self.robustness_evaluator.evaluate_robustness(self.model, self.test_set, self.model_type)

                    # Update score_board and tape_of_moves
                    score_board.append(robust_preservation)
                    print(bcolors.OKGREEN + "[Epoch " + str(epoch_couter) + "]" + str(robust_preservation) + bcolors.ENDC)

                loss, accuracy = self.evaluate(verbose=1)
                accuracy_board.append((round(loss, 4), round(accuracy, 4)))
                    
                tape_of_moves.append(pruned_pairs)
            # Check if have pruned enough number of hidden units
            if self.sampler.mode == SamplingMode.BASELINE and percentage_been_pruned >= 0.5:
                print(" >> Maximum pruning percentage has been reached")
                stop_condition = True
            elif not stop_condition and percentage_been_pruned >= self.pruning_target:
                print(" >> Target pruning percentage has been reached")
                stop_condition = True

        # Stop elapsed time counting
        end_time = time.time()
        print("Elapsed time: ", round((end_time - start_time)/60.0, 3), "minutes /", int(end_time - start_time), "seconds")

        ################################################################
        # Save the tape of moves                                       #
        ################################################################
        
        # Obtain a timestamp
        local_time = time.localtime()
        timestamp = time.strftime('%b-%d-%H%M', local_time)

        tape_filename = "paoding/logs/" + model_name + "-" + timestamp + "-" + str(self.evaluation_batch)
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
            if evaluator is not None:
                csv_line = [str(eps) for eps in self.target_adv_epsilons]
            else:
                csv_line = []
            csv_line.append('moves,loss,accuracy')
            csv_writer.writerow(csv_line)

            
            for index, item in enumerate(accuracy_board):
                rob_pres_stat = []
                if evaluator is not None:
                    rob_res = score_board[index]
                    for k in self.target_adv_epsilons:
                        rob_pres_stat.append(rob_res[k])
                
                rob_pres_stat.append(tape_of_moves[index])
                rob_pres_stat.append(accuracy_board[index])
                csv_writer.writerow(rob_pres_stat)
                
            
            if evaluator is None:
                csv_writer.writerow(["Elapsed time: ", round((end_time - start_time) / 60.0, 3), "minutes /", int(end_time - start_time), "seconds"])

        if pruned_pairs_all_steps is None:
            self.save_model(pruned_model_path)
        elif self.surgery_mode is True: 
            # final_model_path = self.model_path+"_pruned_surgery"
            final_model_path =pruned_model_path
            self.model = surgeon.create_pruned_model(self.model, pruned_pairs_all_steps, final_model_path, optimizer=self.optimizer, loss_fn=self.loss)
        else:
            print(self.model.summary())
        print("FC pruning accomplished")

    def prune_cnv(self, evaluator=None, save_file=False, pruned_model_path=None, verbose=0):
        if evaluator is not None:
            self.robustness_evaluator = evaluator
            self.target_adv_epsilons = evaluator.epsilons
            self.evaluation_batch = evaluator.batch_size

        utils.create_dir_if_not_exist("paoding/logs/")
        # utils.create_dir_if_not_exist("paoding/save_figs/")
        
        if save_file and pruned_model_path is None:
            pruned_model_path=self.model_path

        # Start elapsed time counting
        start_time = time.time()
        pruning_result_dict = self.sampler.nominate_conv(self.model, prune_percentage=self.pruning_target)

        self.model = pruning_result_dict['model']

        self.model.compile(optimizer= self.optimizer, loss=self.loss,
                      metrics=['accuracy'])

        print("CONV pruning accomplished")

        if self.test_set is not None:
            self.evaluate(verbose=1)
        
        if evaluator is not None and self.test_set is not None:                    
            robust_preservation = self.robustness_evaluator.evaluate_robustness(self.model, self.test_set, self.model_type)
        
        if save_file and os.path.exists(pruned_model_path):
            shutil.rmtree(pruned_model_path)
            print("Overwriting existing pruned model ...")

        if save_file:
            self.model.save(pruned_model_path)
            print(" >>> Pruned model saved")
        else:
            print(" >>> Pruned model won't be saved unless you set \"save_file\" True")
            
        # Stop elapsed time counting
        end_time = time.time()
        print("Elapsed time: ", round((end_time - start_time)/60.0, 3), "minutes /", int(end_time - start_time), "seconds")

        print("Pruning accomplished")
    
    def prune_cnv_step(self, evaluator=None, save_file=False, pruned_model_path=None, verbose=0):
        if evaluator is not None:
            self.robustness_evaluator = evaluator
            self.target_adv_epsilons = evaluator.epsilons
            self.evaluation_batch = evaluator.batch_size

        utils.create_dir_if_not_exist("paoding/logs/")
        # utils.create_dir_if_not_exist("paoding/save_figs/")
        
        if save_file and pruned_model_path is None:
            pruned_model_path=self.model_path+"_conv_pruned"

        # Start elapsed time counting
        start_time = time.time()
        pruning_result_dict = self.sampler.nominate_conv(self.model, prune_percentage=self.pruning_step)

        self.model = pruning_result_dict['model']

        self.model.compile(optimizer= self.optimizer, loss=self.loss,
                      metrics=['accuracy'])

        print("CONV pruning accomplished")

        if self.test_set is not None:
            self.evaluate(verbose=1)
        
        if evaluator is not None and self.test_set is not None:                    
            robust_preservation = self.robustness_evaluator.evaluate_robustness(self.model, self.test_set, self.model_type)
        
        if save_file and os.path.exists(pruned_model_path):
            shutil.rmtree(pruned_model_path)
            print("Overwriting existing pruned model ...")

        if save_file:
            self.model.save(pruned_model_path)
            print(" >>> Pruned model saved")
        else:
            print(" >>> Pruned model won't be saved unless you set \"save_file\" True")
            
        # Stop elapsed time counting
        end_time = time.time()
        print("Elapsed time: ", round((end_time - start_time)/60.0, 3), "minutes /", int(end_time - start_time), "seconds")

        return self.model
    
    
    def gc(self):
        self.path = None
        self.test_set=None
        self.target=0.5
        self.step=0.025
        self.sample_strategy=None 
        self.input_interval=(0,1)
        self.model_type=ModelType.XRAY 
        self.seed_val=None
 