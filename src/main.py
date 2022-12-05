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
        aggregator_name = config.server.aggregator['name']
        #if aggregator_name == 'Krum' and config.server.aggregator['args']['byz'] < 0.5:
        #    aggregator_name = 'MultiKrum'
        model_filename = config.environment.load_model + "_" + aggregator_name   
        pretrained_model_path = os.path.join("save_fl_models", model_filename)
        if config.environment.paoding:
            pretrained_model_path += "_paoding"
        if os.path.exists(pretrained_model_path):
            print(" >> Found a pretrained model (" + pretrained_model_path + "), skip the first benign training rounds.")
            config.environment.pretrain = 1
            model = tf.keras.models.load_model(pretrained_model_path) # Load with weights
            save_model(model, filename=temp_filename)
            return model
    
    print(" >> Pretrained model (" + pretrained_model_path + ") not found. Creating a new one...")
    config.environment.pretrain = 0
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
        print(" >> Error while deleting file ", temp_filename)

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
    
    DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS = 0.2 # 20% OUT OF ALL CLIENTS （See Fang's paper, 20% is the worst case in their experiment)
    DEFAULT_ATT_FREQ = 1
    DEFAULT_NUM_MALICIOUS_CLIENTS = int(config.environment.num_selected_clients * DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS) 

    config.environment.attacker_full_knowledge = False
    #config.server.num_rounds = 105
    config.environment.num_malicious_clients = DEFAULT_NUM_MALICIOUS_CLIENTS
    config.environment.attack_frequency = DEFAULT_ATT_FREQ
    config.environment.prune_frequency = 0.25
    config.environment.paoding = 1
    
    tm_beta_list = [0.1,0.2,0.3]
    byz_list = [0.1,0.2,0.3]
    
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
    DEFAULT_REPEAT = 4
    MODE = 'B'

    RQ0 = 1
    RQ1 = 1
    RQ2 = 1
    RQ3 = 1

    if MODE == 'A':
        config.environment.save_model_at = []
        #config.server.num_rounds = 30
        config.environment.load_model = None
    #else:
        #config.client.malicious.attack_stop = 30
        #config.server.num_rounds = 35
        #config.environment.save_model_at = [20]
        #config.server.num_rounds = 30

    if RQ0:
        config.environment.attacker_full_knowledge = False
        config.environment.num_malicious_clients = 0 
        config.environment.attack_frequency = 0.0001
        for paoding_option in [0]:
            config.environment.paoding = paoding_option

            # Step 1: FedAvg pretraining
            
            config.server.aggregator['name'] = 'FedAvg'
            curr_exp_settings = []
            curr_exp_settings.append(config.dataset.dataset)
            curr_exp_settings.append('RQ0-Benign')
                
            if paoding_option == 1:
                curr_exp_settings.append('paoding')
                
            log_filename = generate_logfile_name(curr_exp_settings)

            print(experiment_name + "(RQ0 Pre-train, FedAvg) started.") 
            try:
                main(config, pruning_settings, log_filename)
                        
            except Exception as err:
                print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

  
            # Step 2: TrimmedMean pretraining
            config.server.aggregator['name'] = 'TrimmedMean'
            for tm_beta in tm_beta_list:
                config.server.aggregator['args']['beta']=tm_beta

                curr_exp_settings = []
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ0-Benign')
                
                if tm_beta > DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                    curr_exp_settings.append("TrimMean-Radi")
                elif tm_beta == DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                    curr_exp_settings.append("TrimMean-Perfect")
                else:
                    curr_exp_settings.append("TrimMean-Cons")
                    
                    
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')

                log_filename = generate_logfile_name(curr_exp_settings)
                print(experiment_name + "(RQ0 Pre-train, TM) started.") 
                try:
                    main(config, pruning_settings, log_filename)
                except Exception as err:
                    print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
            
            config.server.aggregator['args'].pop('beta', None)
            
            # Step 3: Krum & Multi-Krum pretraining
            config.server.aggregator['name'] = 'Krum'
            for byz in byz_list:
                config.server.aggregator['args']['byz']=byz
                curr_exp_settings = []
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ0-Benign')
                if byz < 0.5:
                    curr_exp_settings.append("Multi-Krum")
                    if byz > DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                        curr_exp_settings.append("Radi")
                    elif byz == DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                        curr_exp_settings.append("Perfect")
                    else:
                        curr_exp_settings.append("Cons")
                else:
                    curr_exp_settings.append("Krum")
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')

                log_filename = generate_logfile_name(curr_exp_settings)
                print(experiment_name + "(RQ0 Pre-train, Krum & Multi-Krum) started.") 
                
                try:
                    main(config, pruning_settings, log_filename)
                except Exception as err:
                    print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    
            config.server.aggregator['args'].pop('byz', None)

    if RQ1:
        config.environment.attacker_full_knowledge = False
        config.environment.num_malicious_clients = DEFAULT_NUM_MALICIOUS_CLIENTS 
        config.environment.attack_frequency = DEFAULT_ATT_FREQ

        config.server.aggregator['name'] = 'FedAvg'
        #list_of_attack_freq = [0.04, 0.2, 1]
        #list_of_malicious_clients_percentage = [0.0125, 0.1, 0.2, 0.3]
        list_of_attack_freq = [0.0001]
        list_of_malicious_clients_percentage = [0.2]
        
        ## Exp 1. Adjust attack frequency (0.001 means no attack, 0.03 means only 1 attack)
        for attack_freq in list_of_attack_freq: 
            config.environment.attack_frequency = attack_freq
            for paoding_option in [0,1]:
                config.environment.paoding = paoding_option

                curr_exp_settings = []
                curr_exp_settings.append(str(exp_idx) + MODE)
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ1b')
                curr_exp_settings.append(str(attack_freq))
                if config.client.malicious.multi_attacker_scale_divide:
                    curr_exp_settings.append("Super")
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')
                
                if exp_idx < RESUME:
                    print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ1 Freq) skipped.")                    
                else:
                    log_filename = generate_logfile_name(curr_exp_settings)
                    for i in range(0, DEFAULT_REPEAT):
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ1 Freq) started.") 
                        print("  currently in a repeation (" + str(i+1) + "/" + str(DEFAULT_REPEAT) + ")")
                        '''
                        main(config, pruning_settings, log_filename)
                        '''
                        try:
                            main(config, pruning_settings, log_filename)
                        
                        except Exception as err:
                            print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)
                        
                exp_idx += 1

        ## Exp 2. Adjust malicious clients (excluding default mode (15%))
        config.environment.num_malicious_clients = DEFAULT_NUM_MALICIOUS_CLIENTS 
        config.environment.attack_frequency = DEFAULT_ATT_FREQ

        for num_malicious_percentage in list_of_malicious_clients_percentage:        
            config.environment.num_malicious_clients = int(num_malicious_percentage * config.environment.num_selected_clients) 
            #config.client.malicious.backdoor['tasks'] = config.environment.num_malicious_clients
            #config.client.malicious.backdoor['tasks'] = int(num_malicious_percentage * config.environment.num_selected_clients) 
            for paoding_option in [0,1]:
                config.environment.paoding = paoding_option

                curr_exp_settings = []
                curr_exp_settings.append(str(exp_idx) + MODE)
                curr_exp_settings.append(config.dataset.dataset)
                curr_exp_settings.append('RQ1b')
                curr_exp_settings.append(str(config.environment.num_malicious_clients)+"-attcker")
                if config.client.malicious.multi_attacker_scale_divide:
                    curr_exp_settings.append("Super")
                if paoding_option == 1:
                    curr_exp_settings.append('paoding')
                
                if exp_idx < RESUME:
                    print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ1 # Clients) skipped.")                    
                else:
                    log_filename = generate_logfile_name(curr_exp_settings)
                    for i in range(0, DEFAULT_REPEAT):
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ1 # Clients) started.") 
                        print("  currently in a repeation (" + str(i+1) + "/" + str(DEFAULT_REPEAT) + ")")
                        
                        try:
                            main(config, pruning_settings, log_filename)
                        except Exception as err:
                            print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                            print(exc_type, fname, exc_tb.tb_lineno)
                        
                exp_idx += 1
        

    if RQ2:
        reject_options = ['None','ERR', 'LFR', 'UNION']
        
        config.environment.attacker_full_knowledge = False
        config.environment.num_malicious_clients = DEFAULT_NUM_MALICIOUS_CLIENTS 
        config.environment.attack_frequency = DEFAULT_ATT_FREQ
        # Reset task number
        #config.client.malicious.backdoor['tasks'] = config.environment.num_malicious_clients
        
        for reject in reject_options:
            for tm_beta in tm_beta_list:
                config.server.aggregator['name'] = 'TrimmedMean'
                config.server.aggregator['args']['beta']=tm_beta
                config.environment.reject = reject
                for paoding_option in [0,1]:
                    config.environment.paoding = paoding_option

                    curr_exp_settings = []
                    curr_exp_settings.append(str(exp_idx) + MODE)
                    curr_exp_settings.append(config.dataset.dataset)
                    curr_exp_settings.append('RQ2b')
                    if tm_beta > DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                        curr_exp_settings.append("TrimMean-Radi")
                    elif tm_beta == DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                        curr_exp_settings.append("TrimMean-Perfect")
                    else:
                        curr_exp_settings.append("TrimMean-Cons")
                    
                    if reject != 'None':
                        curr_exp_settings.append(reject)
                    if config.client.malicious.multi_attacker_scale_divide:
                        curr_exp_settings.append("Super")    
                    if paoding_option == 1:
                        curr_exp_settings.append('paoding')

                    if exp_idx < RESUME:
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ2 TM) skipped.")                    
                    else:
                        log_filename = generate_logfile_name(curr_exp_settings)
                        for i in range(0, DEFAULT_REPEAT):
                            print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ2 TM) started.") 
                            print("  currently in a repeation (" + str(i+1) + "/" + str(DEFAULT_REPEAT) + ")")
                            
                            try:
                                main(config, pruning_settings, log_filename)
                            except Exception as err:
                                print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                print(exc_type, fname, exc_tb.tb_lineno)
                            
                    exp_idx += 1

        for reject in reject_options:
            config.server.aggregator['name'] = 'Krum'
            config.server.aggregator['args'].pop('beta', None)
            config.environment.reject = reject
            for byz in byz_list:
                config.server.aggregator['args']['byz']=byz
                for paoding_option in [0,1]:
                    config.environment.paoding = paoding_option
                    curr_exp_settings = []
                    curr_exp_settings.append(str(exp_idx) + MODE)
                    curr_exp_settings.append(config.dataset.dataset)
                    curr_exp_settings.append('RQ2b')
                    if byz < 0.5:
                        curr_exp_settings.append("Multi-Krum")
                        if byz > DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                            curr_exp_settings.append("Radi")
                        elif byz == DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                            curr_exp_settings.append("Perfect")
                        else:
                            curr_exp_settings.append("Cons")
                    else:
                        curr_exp_settings.append("Krum")
                    
                    if reject != 'None':
                        curr_exp_settings.append(reject)
                    if config.client.malicious.multi_attacker_scale_divide:
                        curr_exp_settings.append("Super")
                    if paoding_option == 1:
                        curr_exp_settings.append('paoding')

                    if exp_idx < RESUME:
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ2 Krum) skipped.")                    
                    else:
                        log_filename = generate_logfile_name(curr_exp_settings)
                        for i in range(0, DEFAULT_REPEAT):
                            print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ2 Krum) started.") 
                            print("  currently in a repeation (" + str(i+1) + "/" + str(DEFAULT_REPEAT) + ")")
                            try:
                                main(config, pruning_settings, log_filename)
                            
                            except Exception as err:
                                print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                print(exc_type, fname, exc_tb.tb_lineno)
                            
                    exp_idx += 1
        config.server.aggregator['args'].pop('byz', None)

    if RQ3:
        reject_options = ['None', 'ERR', 'LFR', 'UNION']

        config.environment.num_malicious_clients = DEFAULT_NUM_MALICIOUS_CLIENTS 
        config.environment.attack_frequency = DEFAULT_ATT_FREQ
        # Reset task number
        #config.client.malicious.backdoor['tasks'] = config.environment.num_malicious_clients
        
        config.environment.attacker_full_knowledge = True
        
        for reject in reject_options:
            for attacker_full_dataset in [False]:
                config.environment.attacker_full_dataset = attacker_full_dataset
                
                config.server.aggregator['args'].pop('byz', None)
                config.server.aggregator['args'].pop('beta', None)
                config.environment.reject = reject
                '''
                for paoding_option in [0,1]:
                    config.server.aggregator['name'] = 'FedAvg'
                    config.environment.paoding = paoding_option

                    curr_exp_settings = []
                    curr_exp_settings.append(str(exp_idx) + MODE)
                    curr_exp_settings.append(config.dataset.dataset)
                    curr_exp_settings.append('RQ3b')
                    if attacker_full_dataset:
                        curr_exp_settings.append("FK")
                    else:
                        curr_exp_settings.append("PK")
                    curr_exp_settings.append("FedAvg")
                    
                    if reject != 'None':
                        curr_exp_settings.append(reject)
                    if config.client.malicious.multi_attacker_scale_divide:
                        curr_exp_settings.append("Super")
                    if paoding_option == 1:
                        curr_exp_settings.append('paoding')
                        
                    if exp_idx < RESUME:
                        print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 FedAvg) skipped.")                    
                    else:
                        log_filename = generate_logfile_name(curr_exp_settings)
                        for i in range(0, DEFAULT_REPEAT):
                            print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 FedAvg) started.") 
                            print("  currently in a repeation (" + str(i+1) + "/" + str(DEFAULT_REPEAT) + ")")
                            
                            try:
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
                    config.environment.reject = reject
                    for paoding_option in [0,1]:
                        config.environment.paoding = paoding_option
                        curr_exp_settings = []
                        curr_exp_settings.append(str(exp_idx) + MODE)
                        curr_exp_settings.append(config.dataset.dataset)
                        curr_exp_settings.append('RQ3b')
                        if attacker_full_dataset:
                            curr_exp_settings.append("FK")
                        else:
                            curr_exp_settings.append("PK")
                        
                        if tm_beta > DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                            curr_exp_settings.append("TrimMean-Radi")
                        elif tm_beta == DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                            curr_exp_settings.append("TrimMean-Perfect")
                        else:
                            curr_exp_settings.append("TrimMean-Cons")

                        if reject != 'None':
                            curr_exp_settings.append(reject)
                        if config.client.malicious.multi_attacker_scale_divide:
                            curr_exp_settings.append("Super")
                        if paoding_option == 1:
                            curr_exp_settings.append('paoding')
                        
                        if exp_idx < RESUME:
                            print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 TM) skipped.")                    
                        else:
                            log_filename = generate_logfile_name(curr_exp_settings)
                            for i in range(0, DEFAULT_REPEAT):
                                print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 TM) started.") 
                                print("  currently in a repeation (" + str(i+1) + "/" + str(DEFAULT_REPEAT) + ")")
                                
                                try:
                                    main(config, pruning_settings, log_filename)
                                except Exception as err:
                                    print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                                    exc_type, exc_obj, exc_tb = sys.exc_info()
                                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                    print(exc_type, fname, exc_tb.tb_lineno)
                                
                        exp_idx += 1
                
                config.server.aggregator['args'].pop('beta', None)
                config.server.aggregator['name'] = 'Krum'
                config.environment.reject = reject
                for byz in byz_list:
                    config.server.aggregator['args']['byz']=byz
                    for paoding_option in [0,1]:
                        config.environment.paoding = paoding_option
                        curr_exp_settings = []
                        curr_exp_settings.append(str(exp_idx) + MODE)
                        curr_exp_settings.append(config.dataset.dataset)
                        curr_exp_settings.append('RQ3b')
                        if attacker_full_dataset:
                            curr_exp_settings.append("FK")
                        else:
                            curr_exp_settings.append("PK")
                        
                        if byz < 0.5:
                            curr_exp_settings.append("Multi-Krum")
                            if byz > DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                                curr_exp_settings.append("Radi")
                            elif byz == DEFAULT_PERCENTAGE_OF_MALICIOUS_CLIENTS:
                                curr_exp_settings.append("Perfect")
                            else:
                                curr_exp_settings.append("Cons")
                        else:
                            curr_exp_settings.append("Krum")
                        
                        if reject != 'None':
                            curr_exp_settings.append(reject)
                        if config.client.malicious.multi_attacker_scale_divide:
                            curr_exp_settings.append("Super")
                        if paoding_option == 1:
                            curr_exp_settings.append('paoding')
                        
                        if exp_idx < RESUME:
                            print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 Krum) skipped.")                    
                        else:
                            log_filename = generate_logfile_name(curr_exp_settings)
                            for i in range(0, DEFAULT_REPEAT):
                                print(experiment_name + " Experiment no." + str(exp_idx) + " (RQ3 Krum) started.") 
                                print("  currently in a repeation (" + str(i+1) + "/" + str(DEFAULT_REPEAT) + ")")
                                
                                try:
                                    main(config, pruning_settings, log_filename)
                                except Exception as err:
                                    print("An exception occurred in experiment no." + str(exp_idx) + ": " + str(err))
                                    exc_type, exc_obj, exc_tb = sys.exc_info()
                                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                    print(exc_type, fname, exc_tb.tb_lineno)
                                
                        exp_idx += 1
            config.server.aggregator['args'].pop('byz', None)
                    