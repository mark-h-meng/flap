from sqlite3 import Timestamp
import tensorflow as tf
import numpy as np
import time 

from src.client_attacks import Attack
from src.config_cli import get_config
from src.federated_averaging import FederatedAveraging
from src.tf_model import Model
from src.config.definitions import Config

import logging

logger = logging.getLogger(__name__)

def load_model():
    if config.environment.load_model is not None:
        model = tf.keras.models.load_model(config.environment.load_model) # Load with weights
    else:
        model = Model.create_model(
            config.client.model_name, config.server.intrinsic_dimension,
            config.client.model_weight_regularization, config.client.disable_bn)

    save_model(model)
    return model

def save_model(model):
    weights = np.concatenate([x.flatten() for x in model.get_weights()])
    #np.savetxt("resnet18_intrinsic_40k.txt", weights)
    np.savetxt("temp_model.txt", weights)

def main(config, pruning_settings, log_filename):
    models = [load_model()]
    
    # Log some critical info of the current training
    #with open(log_filename, "a") as myfile:
    #    myfile.write(str(config.environment) + "\n")
    #    myfile.write(str(config.server.aggregator) + "\n")
    #    myfile.write(str(config.client.malicious.backdoor) + "\n")

    if config.client.malicious is not None:
        config.client.malicious.attack_type = Attack.UNTARGETED.value \
                         if config.client.malicious.objective['name'] == "UntargetedAttack" else Attack.BACKDOOR.value

    server_model = FederatedAveraging(config, models, args.config_filepath)
    server_model.init()

    start_time = time.time()
    
    server_model.fit(pruning=config.environment.paoding, log_file=log_filename, pruning_settings=pruning_settings)
    
    end_time = time.time()

    with open(log_filename, "a") as myfile:
        myfile.write("Elapsed time: " + str(end_time - start_time) + "\n")
    return

def generate_logfile_name(curr_exp_settings=[]):
    local_time = time.localtime()
    timestamp = time.strftime('%b-%d-%H%M', local_time)
    
    log_filename = "logs/" + config.client.model_name + "-" + timestamp 
    if len(curr_exp_settings) > 0:
        setting_str = '-'.join(map(str, curr_exp_settings))
        log_filename += "-"
        log_filename += setting_str
    
    log_filename += ".csv"
    return log_filename

### Now let's try to run it in batch, by rewriting the main function

if __name__ == '__main__':
    config: Config
    config, args = get_config()
    np.random.seed(config.environment.seed)
    tf.random.set_seed(config.environment.seed)
    # Now let double confirm the default configurations

    DEFAULT_NUM_MAL_WORKDERS = 9

    config.server.aggregator['name'] = 'FedAvg'
    config.server.aggregator['args'] = {}
    
    config.environment.attacker_full_knowledge = False
    config.server.num_rounds = 25
    config.environment.num_malicious_clients = DEFAULT_NUM_MAL_WORKDERS # 30% OUT OF 30 CLIENTS
    config.environment.attack_frequency = 0.5
    config.environment.paoding = 1
    config.environment.prune_frequency = 1
    
    pruning_evaluation_type = 'mnist'
    if config.dataset.dataset=='cifar10':
        pruning_evaluation_type = 'cifar'
    pruning_target = 0.01
    pruning_step = 0.01
    pruning_settings = (pruning_target, pruning_step, pruning_evaluation_type)

    RESUME = 0
    DEFAULT_REPEAT = 5
    BASE_SETTINGS = 0
    RQ1 = 1
    RQ2 = 0
    RQ3 = 0
    # Now we perform a series of experiments by adjusting certain settings
    exp_idx = 0
    if RQ1:
        config.environment.attacker_full_knowledge = False
        config.server.num_rounds = 25
        config.environment.num_malicious_clients = DEFAULT_NUM_MAL_WORKDERS # 30% OUT OF 30 CLIENTS
        config.environment.attack_frequency = 0.5
        config.environment.prune_frequency = 1
        
         
        config.environment.num_malicious_clients = DEFAULT_NUM_MAL_WORKDERS # 30% OUT OF 30 CLIENTS
        config.environment.attack_frequency = 0.5
        config.environment.prune_frequency = 1

        for num_malicious in [3]:
            config.environment.attacker_full_knowledge = False
            config.server.num_rounds = 25
            config.environment.num_malicious_clients = DEFAULT_NUM_MAL_WORKDERS # 30% OUT OF 30 CLIENTS
            config.environment.attack_frequency = 0.5
            config.environment.paoding = 1
            config.environment.prune_frequency = 1
            
            config.environment.num_malicious_clients = num_malicious
            config.client.malicious.backdoor['tasks'] = num_malicious
            for paoding_option in [1]:
                config.environment.paoding = paoding_option
                curr_exp_settings = []
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ1')
                curr_exp_settings.append(str(num_malicious)+"-attcker")
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')
                elif paoding_option == 2:
                    curr_exp_settings.append('adaptive')
                
                if exp_idx < RESUME:
                    print("Experiment no." + str(exp_idx) + " skipped.")
                else:
                    log_filename = generate_logfile_name(curr_exp_settings)
                    for i in range(0, DEFAULT_REPEAT):
                        main(config, pruning_settings, log_filename)
                                           
                exp_idx += 1
    if RQ2:
        config.environment.attacker_full_knowledge = False
        config.server.num_rounds = 25
        config.environment.num_malicious_clients = DEFAULT_NUM_MAL_WORKDERS # 30% OUT OF 30 CLIENTS
        config.environment.attack_frequency = 0.5
        config.environment.prune_frequency = 1
        
        for paoding_option in [0,1]:
            config.environment.paoding = paoding_option
            curr_exp_settings = []
            curr_exp_settings.append(config.dataset.dataset)
            curr_exp_settings.append('RQ2')
            curr_exp_settings.append("FedAvg")
            if paoding_option == 1:
                curr_exp_settings.append('paoding')
            elif paoding_option == 2:
                curr_exp_settings.append('adaptive')
                
            if exp_idx < RESUME:
                print("Experiment no." + str(exp_idx) + " skipped.")
            else:
                log_filename = generate_logfile_name(curr_exp_settings)
                for i in range(0, DEFAULT_REPEAT):
                    try:
                        print("Experiment no." + str(exp_idx) + " started.") 
                        main(config, pruning_settings, log_filename)
                    except:
                        print("An exception occurred in experiment no." + str(exp_idx))                   
            exp_idx += 1
        
        for tm_beta in [0.1, 0.4]:
            config.server.aggregator['name'] = 'TrimmedMean'
            config.server.aggregator['args']['beta']=tm_beta
            for paoding_option in [0,1]:
                config.environment.paoding = paoding_option
                curr_exp_settings = []
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ2')
                if tm_beta > 0.25:
                    curr_exp_settings.append("TrimMean-Radi")
                else:
                    curr_exp_settings.append("TrimMean-Cons")
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')
                elif paoding_option == 2:
                    curr_exp_settings.append('adaptive')
                    
                if exp_idx < RESUME:
                    print("Experiment no." + str(exp_idx) + " skipped.")
                else:
                    log_filename = generate_logfile_name(curr_exp_settings)
                    for i in range(0, DEFAULT_REPEAT):
                        try:
                            print("Experiment no." + str(exp_idx) + " started.") 
                            main(config, pruning_settings, log_filename)
                        except:
                            print("An exception occurred in experiment no." + str(exp_idx))                   
                exp_idx += 1

        config.server.aggregator['name'] = 'Krum'
        config.server.aggregator['args'].pop('beta', None)
        config.server.aggregator['args']['byz']=0.33
        for paoding_option in [0,1]:
            config.environment.paoding = paoding_option
            curr_exp_settings = []
            curr_exp_settings.append(config.dataset.dataset)
            curr_exp_settings.append('RQ2')
            curr_exp_settings.append("Krum")
            if paoding_option == 1:
                curr_exp_settings.append('paoding')
            elif paoding_option == 2:
                curr_exp_settings.append('adaptive')
                    
            if exp_idx < RESUME:
                print("Experiment no." + str(exp_idx) + " skipped.")
            else:
                log_filename = generate_logfile_name(curr_exp_settings)
                for i in range(0, DEFAULT_REPEAT):
                    try:
                        print("Experiment no." + str(exp_idx) + " started.") 
                        main(config, pruning_settings, log_filename)
                    except:
                        print("An exception occurred in experiment no." + str(exp_idx))                   
            exp_idx += 1        
    if RQ3:
        config.server.num_rounds = 25
        config.environment.num_malicious_clients = DEFAULT_NUM_MAL_WORKDERS # 30% OUT OF 30 CLIENTS
        config.environment.attack_frequency = 0.5
        config.environment.prune_frequency = 1
        
        config.environment.attacker_full_knowledge = True
        for attacker_full_dataset in [True]:
            config.environment.attacker_full_dataset = attacker_full_dataset
            
            
            for tm_beta in [0.1]:
                config.server.aggregator['name'] = 'TrimmedMean'
                config.server.aggregator['args']['beta']=tm_beta
                for paoding_option in [1]:
                    config.environment.paoding = paoding_option
                    curr_exp_settings = []
                    curr_exp_settings.append('RQ3')
                    curr_exp_settings.append(config.dataset.dataset)
                    if attacker_full_dataset:
                        curr_exp_settings.append("FullKn")
                    else:
                        curr_exp_settings.append("PartialKn")
                    if tm_beta > 0.25:
                        curr_exp_settings.append("TrimMean-Radi")
                    else:
                        curr_exp_settings.append("TrimMean-Cons")
                    if paoding_option == 1:
                        curr_exp_settings.append('paoding')
                        
                    if exp_idx < RESUME:
                        print("Experiment no." + str(exp_idx) + " skipped.")
                    else:
                        log_filename = generate_logfile_name(curr_exp_settings)
                        for i in range(0, DEFAULT_REPEAT):
                            try:
                                print("Experiment no." + str(exp_idx) + " started.") 
                                main(config, pruning_settings, log_filename)
                            except:
                                print("An exception occurred in experiment no." + str(exp_idx))                   
                    exp_idx += 1
            
            config.server.aggregator['args'].pop('beta', None)
            