#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

# Import publicly published & installed packages
import tensorflow as tf
import time

# Import in-house classes
from paoding.utility.option import SamplingMode
import paoding.utility.pruning as pruning

class Sampler:

    mode = -1
    mode_conv = -1
    params=(0, 0)
    recursive_pruning = False

    def __init__(self, mode=SamplingMode.BASELINE, recursive_pruning=False):
        """Initializes `Sampler` class.
        Args:
        mode: The mode of sampling strategy (optional, baseline mode by default).
            [PS] 3 modes are supported in the Alpha release, refer to the ``paoding.utility.option.SamplingMode`` for the technical definition.
        """
        self.mode = mode
        self.mode_conv = SamplingMode.SCALE
        self.recursive_pruning = recursive_pruning
    

    def set_strategy(self, mode, params=(0.75, 0.25), recursive_pruning=False):
        """Set the sampling strategy.
        Args:
        mode: The mode of sampling strategy (optional, baseline mode by default).
            [PS] 3 modes are supported in the Alpha release, refer to the ``paoding.utility.option.SamplingMode`` for the technical definition.
        params: The tuple of parameters (for greedy and stochastic modes only) (optional, (0.75, 0.25) by default).
        """
        self.mode = mode
        self.mode_conv = SamplingMode.SCALE
        self.params = params
        self.recursive_pruning = recursive_pruning


    def nominate(self, model, big_map, prune_percentage=0.5,
                     neurons_manipulated=None, saliency_matrix=None,
                     cumulative_impact_intervals=None,
                     bias_aware=False, pooling_multiplier=2,
                     target_scores=None, verbose=0):
        """
        Nominate and prune the hidden units.
        Args:
        model: The model to be pruned.
        big_map: The matrix of correlation between every hidden unit pairs.
        prune_percentage: The goal of pruning (optional, 0.5 by default).
        neurons_manipulated: The list of hidden unit indices that have been involved in previous pruning operations (optional, None by default).
        saliency_matrix: The matrix of saliency between every hidden unit pairs, only applicable for baseline mode (optional, None by default).
        recursive_pruning: The boolean parameter to indicate if recursive pruning is allowed (i.e., a neuron can be involved in pruning multiple times) (optional, False by default).
        cumulative_impact_intervals: The cumulative pruning impact of all previous pruning operations (optional, None by default).
        bias_aware: The boolean parameter to indicate if bias parameters to be considered in pruning (optional, False by default).
        pooling_multiplier: The sampling multiplier at each pruning epoch, only applicable for stochastic mode (optional, 2 by default).
        target_scores: The impact tolerance observed from the previous pruning operation, only applicable for stochastic mode (optional, None by default).
        Returns:
        A dictionary data structure including the pruned model, the list of neurons that have been manipulated (neurons_manipulated), 
            target score observed from the current pruning operation (target_scores), the list of pairs have been nominated in the 
            current pruning epoch (pruned_pairs), saliency matrix (saliency_matrix), cumulative impact intervals (cumulative_impact_intervals), 
            and the latest assessment of every hidden unit pairs for further pruning (pruning_pairs_dict_overall_scores).
        """
        pruned_pairs = None
        pruning_pairs_dict_overall_scores = None
        if self.mode == SamplingMode.BASELINE:
            result = pruning.pruning_baseline(model, big_map, prune_percentage, neurons_manipulated,
                                        saliency_matrix, self.recursive_pruning, bias_aware)
            (model, neurons_manipulated, pruned_pairs, saliency_matrix) = result

            count_pairs_pruned_curr_epoch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        print(" >> Pruning", pairs, "at layer", str(layer))
                        for pair in pairs:
                            count_pairs_pruned_curr_epoch += 1

        elif self.mode == SamplingMode.GREEDY:
            result = pruning.pruning_greedy(model, big_map, prune_percentage,
                   cumulative_impact_intervals,
                   pooling_multiplier,
                   neurons_manipulated,
                   self.params,
                   self.recursive_pruning,
                   bias_aware,
                   kaggle_credit=False)
            (model, neurons_manipulated, pruned_pairs, cumulative_impact_intervals, pruning_pairs_dict_overall_scores) = result

            count_pairs_pruned_curr_epoch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        if verbose > 0:
                            print(" [DEBUG] Pruning", pairs, "at layer", str(layer))
                            print("      with assessment score ", end=' ')
                        for pair in pairs:
                            count_pairs_pruned_curr_epoch += 1
                            print(round(pruning_pairs_dict_overall_scores[layer][pair], 3), end=' ')
                        print()

        elif self.mode == SamplingMode.STOCHASTIC:
            result = pruning.pruning_stochastic(model, big_map, prune_percentage,
                      cumulative_impact_intervals,
                      neurons_manipulated,
                      target_scores,
                      self.params,
                      self.recursive_pruning,
                      bias_aware,
                      kaggle_credit=False)
            (model, neurons_manipulated, target_scores, pruned_pairs, cumulative_impact_intervals, pruning_pairs_dict_overall_scores) = result

            count_pairs_pruned_curr_epoch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        if verbose > 0:
                            print(" [DEBUG] Pruning", pairs, "at layer", str(layer))
                            print("      with assessment score ", end=' ')
                            for pair in pairs:
                                count_pairs_pruned_curr_epoch += 1
                                print(round(pruning_pairs_dict_overall_scores[layer][pair], 3), end=' ')
                            print()
                            print(" [DEBUG] Updated target scores at this layer:", round(target_scores[layer], 3))
        
        elif self.mode == SamplingMode.SCALE:
            result = pruning.pruning_scale_only_sparse(model, prune_percentage)
            (model) = result

            print(" >> Pruning accomplished.")
        
        else:
            print("Sampling mode not recognized, execution aborted!")
        
        result_dict = {
            'model': model,
            'neurons_manipulated': neurons_manipulated,
            'target_scores': target_scores,
            'pruned_pairs': pruned_pairs,
            'saliency_matrix': saliency_matrix,
            'cumulative_impact_intervals': cumulative_impact_intervals,
            'pruning_pairs_dict_overall_scores': pruning_pairs_dict_overall_scores
        }

        return result_dict

    def nominate_conv(self, model, prune_percentage=0.5):
        """
        Nominate and prune the hidden units in convolutional layers.
        Args:
        model: The model to be pruned.
        big_map: The matrix of correlation between every hidden unit pairs.
        prune_percentage: The goal of pruning (optional, 0.5 by default).
        
        A dictionary data structure including the pruned model.
        """
        if self.mode_conv == SamplingMode.SCALE:
            result = pruning.pruning_conv_scale(model, prune_percentage)
            (model) = result

            result_dict = {
                'model': model
            }

        return result_dict