from copy import deepcopy

import numpy as np
import math
import logging

class Aggregator:
    """ Aggregation behavior """
    def aggregate(self, global_weights, client_weight_list):
        """

        :type client_weight_list: list[np.ndarray]
        """
        raise NotImplementedError("Subclass")
    
    def fedavg(self, global_weights, client_weight_list):
        return self.aggregate(self, global_weights, client_weight_list)

class FedAvg(Aggregator):

    def __init__(self, lr):
        self.lr = lr

    def aggregate(self, global_weights, client_weight_list):
        """Procedure for merging client weights together with `global_learning_rate`."""
        # return deepcopy(client_weight_list[0]) # Take attacker's
        current_weights = global_weights
        new_weights = deepcopy(current_weights)
        # return new_weights
        update_coefficient = self.lr

        for client in range(0, len(client_weight_list)):
            for layer in range(len(client_weight_list[client])):
                new_weights[layer] = new_weights[layer] + \
                                     update_coefficient * (client_weight_list[client][layer] - current_weights[layer])

                if np.isnan(new_weights[layer]).any(): # TODO: Remove
                    print(f"Layer {layer} is NaN!")
                    import sys
                    np.set_printoptions(threshold=sys.maxsize)
                    print(new_weights[layer])
                    print("XX")
                    print(client_weight_list[client][layer])
                    print("XX")
                    print(current_weights[layer])

        return new_weights

    def fedavg(self, global_weights, client_weight_list):
        return self.aggregate(global_weights, client_weight_list)

class TrimmedMean(Aggregator):

    def __init__(self, beta, lr):
        """

        :type beta: float fraction of values to truncate
        """
        self.beta = beta
        self.lr = lr

        assert 0 < self.beta < 1/2, "Beta must be between zero and 1/2!"

    def aggregate(self, global_weights, client_weight_list):
        logging.info("TrimmedMean is selected, beta = "+str(self.beta))
        assert self.beta < 0.5, "Beta must be smaller than 0.5!"

        truncate_count = int(self.beta * len(client_weight_list))
        assert len(client_weight_list) - (truncate_count * 2) > 0, "Must be more clients for a given beta!"

        current_weights = global_weights
        new_weights = deepcopy(current_weights)

        # sort by parameter
        accumulator = [np.zeros([*layer.shape, len(client_weight_list)], layer.dtype) for layer in new_weights]
        for client in range(0, len(client_weight_list)):
            for layer in range(len(client_weight_list[client])):
                accumulator[layer][..., client] = client_weight_list[client][layer] - current_weights[layer]

        for layer in range(len(accumulator)):
            accumulator[layer] = np.sort(accumulator[layer], -1)
            if truncate_count > 0:
                accumulator[layer] = accumulator[layer][..., truncate_count:-truncate_count]
            else:
                logging.warning(f"Beta is too low ({self.beta}), trimming no values which means we effectively take the mean.")
            new_weights[layer] = new_weights[layer] + \
                                 self.lr * np.mean(accumulator[layer], -1) * \
                                 len(client_weight_list) # Multiply by list of clients

        return new_weights

    def fedavg(self, global_weights, client_weight_list):
        """Procedure for merging client weights together with `global_learning_rate`."""
        # return deepcopy(client_weight_list[0]) # Take attacker's
        current_weights = global_weights
        new_weights = deepcopy(current_weights)
        # return new_weights
        update_coefficient = self.lr

        for client in range(0, len(client_weight_list)):
            for layer in range(len(client_weight_list[client])):
                new_weights[layer] = new_weights[layer] + \
                                     update_coefficient * (client_weight_list[client][layer] - current_weights[layer])

                if np.isnan(new_weights[layer]).any(): # TODO: Remove
                    print(f"Layer {layer} is NaN!")
                    import sys
                    np.set_printoptions(threshold=sys.maxsize)
                    print(new_weights[layer])
                    print("XX")
                    print(client_weight_list[client][layer])
                    print("XX")
                    print(current_weights[layer])

        return new_weights
class Median(Aggregator):

    def __init__(self, lr):
        self.lr = lr

    def aggregate(self, global_weights, client_weight_list):
        logging.info("Median is selected")
        
        median_index = math.ceil(len(client_weight_list)/2) -1
        assert len(client_weight_list) - (median_index) >= 0, "Median index is wrongly calculated!"

        current_weights = global_weights
        new_weights = deepcopy(current_weights)

        # sort by parameter
        accumulator = [np.zeros([*layer.shape, len(client_weight_list)], layer.dtype) for layer in new_weights]
        for client in range(0, len(client_weight_list)):
            for layer in range(len(client_weight_list[client])):
                accumulator[layer][..., client] = client_weight_list[client][layer] - current_weights[layer]

        for layer in range(len(accumulator)):
            accumulator[layer] = np.sort(accumulator[layer], -1)
            accumulator[layer] = accumulator[layer][..., median_index:median_index+1]
            
            new_weights[layer] = new_weights[layer] + \
                                 self.lr * np.mean(accumulator[layer], -1) * \
                                 len(client_weight_list) # Multiply by list of clients

        return new_weights
    
    def fedavg(self, global_weights, client_weight_list):
        """Procedure for merging client weights together with `global_learning_rate`."""
        # return deepcopy(client_weight_list[0]) # Take attacker's
        current_weights = global_weights
        new_weights = deepcopy(current_weights)
        # return new_weights
        update_coefficient = self.lr

        for client in range(0, len(client_weight_list)):
            for layer in range(len(client_weight_list[client])):
                new_weights[layer] = new_weights[layer] + \
                                     update_coefficient * (client_weight_list[client][layer] - current_weights[layer])

                if np.isnan(new_weights[layer]).any(): # TODO: Remove
                    print(f"Layer {layer} is NaN!")
                    import sys
                    np.set_printoptions(threshold=sys.maxsize)
                    print(new_weights[layer])
                    print("XX")
                    print(client_weight_list[client][layer])
                    print("XX")
                    print(current_weights[layer])

        return new_weights
class Krum(Aggregator):

    def __init__(self, byz, lr):
        """

        :type byz: float fraction of byzantine workers to tolerant
        """
        self.byz = byz
        self.lr = lr

        #assert 0 < self.byz < 2/3, "Byz must be between zero and 2/3!"

    def aggregate(self, global_weights, client_weight_list):
        logging.info("Krum is selected, byz = "+str(self.byz))
        num_workers = len(client_weight_list)
        # byz specifies the upper bound of the estimated percentage of malicious workers
        num_byzworkers = int(self.byz * num_workers)
        num_selected = max(num_workers - num_byzworkers - 2, 1)
        
        if num_workers - num_byzworkers - 2 <= 1:
            print(" >>> Krum is selected, only 1 local model is adopted in the current round of aggregation.")
        else:
            print(" >>> Multi-Krum is selected. " + str(num_workers - num_byzworkers - 2) + " clients update will be aggregated this round.")
            
        current_weights = global_weights
        new_weights = deepcopy(current_weights)

        gradients = deepcopy(client_weight_list)
        
        for client in range(len(client_weight_list)):
            client_grad = []
            #print("Calculating gradient for client " + str(client) + " :", end="")
            for layer in range(len(client_weight_list[client])):
                #print(str(layer) + " ", end="")
                grad = client_weight_list[client][layer] - current_weights[layer]
                grad /= float(num_workers)
                client_grad.append(grad)
                #print("size of client_grad[-1]:", np.shape(client_grad[-1]))
            #print("")
            gradients[client] = client_grad
        #print("size of gradients:", np.shape(gradients))
        
        # Compute list of scores
        scores = [list() for i in range(num_workers)]
        for i in range(num_workers):
            #score = scores[i]
            for j in range(i + 1, num_workers):
                distance = 0
                diff = np.array(gradients[i]) - np.array(gradients[j])
                for layer in range(len(diff)): 
                    distance += np.linalg.norm(diff[layer])
                if math.isnan(distance):
                    distance = math.inf
                #print("[i]",i,"[j]",j,"dist.",distance)
                #score.append(distance)
                scores[j].append(distance)
                scores[i].append(distance)

        sum_scores = [list() for i in range(num_workers)]
        
        for i in range(num_workers):
            score = scores[i]
            score.sort()
            sum_scores[i] = sum(score[:num_selected])
        # Return the average of the m gradients with the smallest score
        pairs = [(gradients[i], sum_scores[i]) for i in range(num_workers)]
        pairs.sort(key=lambda pair: pair[1])
        #print(pairs)
        result = pairs[0][0]
        #print("result >>", len(result), len(result[0]))
        for i in range(1, num_selected):
            #result += pairs[i][0]
            result = np.add(result, pairs[i][0])
            #print("result >>", len(result), len(result[0]))
        
        #print("num_selected >>", num_selected)
        #print("result >>", len(result), len(result[0]))
        for layer in range(len(result)):   
            #print("processing layer", layer)         
            new_weights[layer] = new_weights[layer] + \
                                 self.lr * np.array(result[layer], dtype=float)* \
                                 len(client_weight_list) 
        return new_weights

    def fedavg(self, global_weights, client_weight_list):
        """Procedure for merging client weights together with `global_learning_rate`."""
        # return deepcopy(client_weight_list[0]) # Take attacker's
        current_weights = global_weights
        new_weights = deepcopy(current_weights)
        # return new_weights
        update_coefficient = self.lr

        for client in range(0, len(client_weight_list)):
            for layer in range(len(client_weight_list[client])):
                new_weights[layer] = new_weights[layer] + \
                                     update_coefficient * (client_weight_list[client][layer] - current_weights[layer])

                if np.isnan(new_weights[layer]).any(): # TODO: Remove
                    print(f"Layer {layer} is NaN!")
                    import sys
                    np.set_printoptions(threshold=sys.maxsize)
                    print(new_weights[layer])
                    print("XX")
                    print(client_weight_list[client][layer])
                    print("XX")
                    print(current_weights[layer])

        return new_weights

def build_aggregator(config):
    aggregator = config.server.aggregator
    lr = config.server.global_learning_rate
    if lr < 0:
        logging.info("Using default global learning rate of n/m")
        lr = config.environment.num_clients / config.environment.num_selected_clients
    else:
        lr = lr

    weight_coefficient = lr / config.environment.num_clients

    from src import aggregation
    cls = getattr(aggregation, aggregator["name"])
    if "args" in aggregator:
        return cls(lr=weight_coefficient, **aggregator["args"])
    else:
        return cls(lr=weight_coefficient)

    # if aggregator.name == "FedAvg":
    #     return FedAvg(weight_coefficient)
    # elif aggregator.name == "TrimmedMean":
    #     return TrimmedMean(config['trimmed_mean_beta'], weight_coefficient)
    # else:
    #     raise NotImplementedError(f"Aggregator {aggregator} not supported!")

