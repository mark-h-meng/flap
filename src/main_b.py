from sqlite3 import Timestamp
import tensorflow as tf
import numpy as np
import time, sys, os

from src.client_attacks import Attack
from src.config_cli import get_config
from src.federated_averaging import FederatedAveraging
from src.tf_model import Model
from src.config.definitions import Config

import logging

logger = logging.getLogger(__name__)

def get_timestamp_str():
    local_time = time.localtime()
    timestamp = time.strftime('%m%d-%H%M', local_time)
    return timestamp

def load_model(temp_filename):
    if config.environment.load_model is not None:        
        pretrained_model_path = os.path.join("save_fl_models", config.environment.load_model)
        if os.path.exists(pretrained_model_path):
            print(" > Found a pretrained model, skip the first benign training rounds.")
            config.environment.pretrain = 1
            model = tf.keras.models.load_model(pretrained_model_path) # Load with weights
            save_model(model, filename=temp_filename)
            return model
    
    model = Model.create_model(
            config.client.model_name, config.server.intrinsic_dimension,
            config.client.model_weight_regularization, config.client.disable_bn)

    save_model(model, filename=temp_filename)
    return model

def save_model(model, filename="temp_model.txt"):
    weights = np.concatenate([x.flatten() for x in model.get_weights()])
    np.savetxt(filename, weights)

def trash_model(temp_filename):
    try:
        print(" > Removing temp file:", temp_filename)
        os.remove(temp_filename)
    except:
        print("Error while deleting file ", temp_filename)

def main(config, pruning_settings, log_filename):
    timestamp = get_timestamp_str()
    temp_filename = "temp_model_" + timestamp + ".txt"

    models = [load_model(temp_filename)]

    if config.client.malicious is not None:
        config.client.malicious.attack_type = Attack.UNTARGETED.value \
                         if config.client.malicious.objective['name'] == "UntargetedAttack" else Attack.BACKDOOR.value

    server_model = FederatedAveraging(config, models, args.config_filepath)
    server_model.init()
    
    server_model.fit(pruning=config.environment.paoding, log_file=log_filename, pruning_settings=pruning_settings)
    
    trash_model(temp_filename)
    return

def generate_logfile_name(curr_exp_settings=[]):
    timestamp = get_timestamp_str()
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

    
    experiment_name = config.client.model_name
    
    config.server.aggregator['name'] = 'FedAvg'
    config.server.aggregator['args'] = {}
    
    DEFAULT_NUM_MALICIOUS_CLIENTS = int(config.environment.num_selected_clients * 0.15) # 15% OUT OF ALL CLIENTS （See Fang's paper, 15% is the worst case in their experiment)
    DEFAULT_ATT_FREQ = 1

    config.environment.attacker_full_knowledge = False
    #config.server.num_rounds = 105
    config.environment.num_malicious_clients = DEFAULT_NUM_MALICIOUS_CLIENTS
    config.environment.attack_frequency = DEFAULT_ATT_FREQ
    config.environment.prune_frequency = 0.5
    config.environment.paoding = 1
    
    tm_beta_list = [0.1]
    # tm_beta_list = [0.1, 0.4]
    byz_list = [0.1]
    # byz_list = [0.33, 0.1]

    # pruning_evaluation_type is only used to define the log file name
    pruning_evaluation_type = 'mnist'
    if config.dataset.dataset=='cifar10':
        pruning_evaluation_type = 'cifar'

    pruning_target = 0.01
    pruning_step = 0.01
    pruning_settings = (pruning_target, pruning_step, pruning_evaluation_type)

    # Now we perform a series of experiments by adjusting certain settings
    exp_idx = 1
    RESUME = 1
    DEFAULT_REPEAT = 1
    
    RQ1 = 1
    RQ2 = 1
    RQ3 = 1

    if RQ1:
        config.environment.attacker_full_knowledge = False
        config.environment.num_malicious_clients = DEFAULT_NUM_MALICIOUS_CLIENTS 
        config.environment.attack_frequency = DEFAULT_ATT_FREQ

        list_of_attack_freq = [0.0001, 0.2, 1]
        list_of_malicious_clients_percentage = [0.05, 0.1, 0.15, 0.3]
        
        ## Exp 1. Adjust attack frequency (0.001 means no attack, 0.03 means only 1 attack)
        for attack_freq in list_of_attack_freq: 
            config.environment.attack_frequency = attack_freq
            for paoding_option in [0,1]:
                config.environment.paoding = paoding_option

                curr_exp_settings = []
                curr_exp_settings.append(str(exp_idx))
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ1b')
                curr_exp_settings.append(str(attack_freq))
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')
                
                if exp_idx < RESUME:
                    print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ1 Freq) skipped.")                    
                else:
                    log_filename = generate_logfile_name(curr_exp_settings)
                    for i in range(0, DEFAULT_REPEAT):
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ1 Freq) started.") 
                        print("  currently in a repeation (" + str(i) + "/" + str(DEFAULT_REPEAT) + ")")
                        #try:
                        main(config, pruning_settings, log_filename)
                        '''
                        except Exception as err:
                            print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)
                        '''
                exp_idx += 1

        ## Exp 2. Adjust malicious clients (excluding default mode (15%))
        config.environment.num_malicious_clients = DEFAULT_NUM_MALICIOUS_CLIENTS 
        config.environment.attack_frequency = DEFAULT_ATT_FREQ

        for num_malicious_percentage in list_of_malicious_clients_percentage:        
            config.environment.num_malicious_clients = int(num_malicious_percentage * config.environment.num_selected_clients) 
            config.client.malicious.backdoor['tasks'] = config.environment.num_malicious_clients
            #config.client.malicious.backdoor['tasks'] = int(num_malicious_percentage * config.environment.num_selected_clients) 
            for paoding_option in [0,1]:
                config.environment.paoding = paoding_option

                curr_exp_settings = []
                curr_exp_settings.append(str(exp_idx))
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ1b')
                curr_exp_settings.append(str(config.environment.num_malicious_clients)+"-attcker")
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')
                
                if exp_idx < RESUME:
                    print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ1 # Clients) skipped.")                    
                else:
                    log_filename = generate_logfile_name(curr_exp_settings)
                    for i in range(0, DEFAULT_REPEAT):
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ1 # Clients) started.") 
                        print("  currently in a repeation (" + str(i) + "/" + str(DEFAULT_REPEAT) + ")")
                        main(config, pruning_settings, log_filename)
                        '''
                        try:
                            main(config, pruning_settings, log_filename)
                        except Exception as err:
                            print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)
                        '''
                exp_idx += 1
        

    if RQ2:

        config.environment.attacker_full_knowledge = False
        config.environment.num_malicious_clients = DEFAULT_NUM_MALICIOUS_CLIENTS 
        config.environment.attack_frequency = DEFAULT_ATT_FREQ
        # Reset task number
        config.client.malicious.backdoor['tasks'] = config.environment.num_malicious_clients
        
        # We skip FedAvg as it is the default setting and has been tested in RQ1
        '''
        for paoding_option in [0,1]:
            config.environment.paoding = paoding_option
            curr_exp_settings = []
            curr_exp_settings.append(config.dataset.dataset)
            curr_exp_settings.append('RQ2b')

            curr_exp_settings.append("FedAvg")
            if paoding_option == 1:
                curr_exp_settings.append('paoding')
            elif paoding_option == 2:
                curr_exp_settings.append('adaptive')
                
            if exp_idx < RESUME:
                print(experiment_name + " Experiment no." + str(exp_idx) + " skipped.")
            else:
                log_filename = generate_logfile_name(curr_exp_settings)
                for i in range(0, DEFAULT_REPEAT):
                    try:
                        print(experiment_name + " Experiment no." + str(exp_idx) + " started.") 
                        main(config, pruning_settings, log_filename)
                    except Exception as err:
                        print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))   
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)               
            exp_idx += 1
        '''    

        for tm_beta in tm_beta_list:
            config.server.aggregator['name'] = 'TrimmedMean'
            config.server.aggregator['args']['beta']=tm_beta
            for paoding_option in [0,1]:
                config.environment.paoding = paoding_option

                curr_exp_settings = []
                curr_exp_settings.append(str(exp_idx))
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ2b')
                if tm_beta > 0.25:
                    curr_exp_settings.append("TrimMean-Radi")
                else:
                    curr_exp_settings.append("TrimMean-Cons")
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')

                if exp_idx < RESUME:
                    print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ2 TM) skipped.")                    
                else:
                    log_filename = generate_logfile_name(curr_exp_settings)
                    for i in range(0, DEFAULT_REPEAT):
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ2 TM) started.") 
                        print("  currently in a repeation (" + str(i) + "/" + str(DEFAULT_REPEAT) + ")")
                        main(config, pruning_settings, log_filename)
                        '''
                        try:
                            main(config, pruning_settings, log_filename)
                        except Exception as err:
                            print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)
                        '''
                exp_idx += 1

        config.server.aggregator['name'] = 'Krum'
        config.server.aggregator['args'].pop('beta', None)
        for byz in byz_list:
            config.server.aggregator['args']['byz']=byz
            for paoding_option in [0,1]:
                config.environment.paoding = paoding_option
                curr_exp_settings = []
                curr_exp_settings.append(str(exp_idx))
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ2b')
                if byz > 0.25:
                    curr_exp_settings.append("Krum-Radi")
                else:
                    curr_exp_settings.append("Krum-Cons")
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')

                if exp_idx < RESUME:
                    print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ2 Krum) skipped.")                    
                else:
                    log_filename = generate_logfile_name(curr_exp_settings)
                    for i in range(0, DEFAULT_REPEAT):
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ2 Krum) started.") 
                        print("  currently in a repeation (" + str(i) + "/" + str(DEFAULT_REPEAT) + ")")
                        #try:
                        main(config, pruning_settings, log_filename)
                        '''
                        except Exception as err:
                            print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)
                        '''
                exp_idx += 1
        config.server.aggregator['args'].pop('byz', None)

    if RQ3:
        config.environment.num_malicious_clients = DEFAULT_NUM_MALICIOUS_CLIENTS 
        config.environment.attack_frequency = DEFAULT_ATT_FREQ
        # Reset task number
        config.client.malicious.backdoor['tasks'] = config.environment.num_malicious_clients
        
        config.environment.attacker_full_knowledge = True
        for attacker_full_dataset in [False,True]:
            config.environment.attacker_full_dataset = attacker_full_dataset
            
            config.server.aggregator['args'].pop('byz', None)
            config.server.aggregator['args'].pop('beta', None)
            for paoding_option in [0,1]:
                config.server.aggregator['name'] = 'FedAvg'
                config.environment.paoding = paoding_option

                curr_exp_settings = []
                curr_exp_settings.append(str(exp_idx))
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ3b')
                if attacker_full_dataset:
                    curr_exp_settings.append("FK")
                else:
                    curr_exp_settings.append("PK")
                curr_exp_settings.append("FedAvg")
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')
                    
                if exp_idx < RESUME:
                    print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 FedAvg) skipped.")                    
                else:
                    log_filename = generate_logfile_name(curr_exp_settings)
                    for i in range(0, DEFAULT_REPEAT):
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 FedAvg) started.") 
                        print("  currently in a repeation (" + str(i) + "/" + str(DEFAULT_REPEAT) + ")")
                        main(config, pruning_settings, log_filename)
                        '''
                        try:
                            main(config, pruning_settings, log_filename)
                        except Exception as err:
                            print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno) 
                        '''
                exp_idx += 1
            
            for tm_beta in tm_beta_list:
                config.server.aggregator['name'] = 'TrimmedMean'
                config.server.aggregator['args']['beta']=tm_beta
                for paoding_option in [0,1]:
                    config.environment.paoding = paoding_option
                    curr_exp_settings = []
                    curr_exp_settings.append(str(exp_idx))
                    curr_exp_settings.append(config.dataset.dataset)
                    curr_exp_settings.append('RQ3b')
                    if attacker_full_dataset:
                        curr_exp_settings.append("FK")
                    else:
                        curr_exp_settings.append("PK")
                    if tm_beta > 0.25:
                        curr_exp_settings.append("TrimMean-Radi")
                    else:
                        curr_exp_settings.append("TrimMean-Cons")
                    if paoding_option == 1:
                        curr_exp_settings.append('paoding')
                    
                    if exp_idx < RESUME:
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 TM) skipped.")                    
                    else:
                        log_filename = generate_logfile_name(curr_exp_settings)
                        for i in range(0, DEFAULT_REPEAT):
                            print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 TM) started.") 
                            print("  currently in a repeation (" + str(i) + "/" + str(DEFAULT_REPEAT) + ")")
                            main(config, pruning_settings, log_filename)
                            '''
                            try:
                                main(config, pruning_settings, log_filename)
                            except Exception as err:
                                print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                print(exc_type, fname, exc_tb.tb_lineno)
                            '''
                    exp_idx += 1
            
            config.server.aggregator['args'].pop('beta', None)
            config.server.aggregator['name'] = 'Krum'
            for byz in byz_list:
                config.server.aggregator['args']['byz']=byz
                for paoding_option in [0,1]:
                    config.environment.paoding = paoding_option
                    curr_exp_settings = []
                    curr_exp_settings.append(str(exp_idx))
                    curr_exp_settings.append(config.dataset.dataset)
                    curr_exp_settings.append('RQ3b')
                    if attacker_full_dataset:
                        curr_exp_settings.append("FK")
                    else:
                        curr_exp_settings.append("PK")
                    if byz > 0.25:
                        curr_exp_settings.append("Krum")
                    else:
                        curr_exp_settings.append("Krum-Cons")
                    if paoding_option == 1:
                        curr_exp_settings.append('paoding')
                    
                    if exp_idx < RESUME:
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 Krum) skipped.")                    
                    else:
                        log_filename = generate_logfile_name(curr_exp_settings)
                        for i in range(0, DEFAULT_REPEAT):
                            print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 Krum) started.") 
                            print("  currently in a repeation (" + str(i) + "/" + str(DEFAULT_REPEAT) + ")")
                            main(config, pruning_settings, log_filename)
                            '''
                            try:
                                main(config, pruning_settings, log_filename)
                            except Exception as err:
                                print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                print(exc_type, fname, exc_tb.tb_lineno)
                            '''
                    exp_idx += 1
        config.server.aggregator['args'].pop('byz', None)
                