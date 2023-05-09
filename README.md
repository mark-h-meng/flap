# FLAP: Federated learning by data-agnostic pruning

### THIS REPO IS IMPLEMENTED BASED ON A PUBLIC PROJECT https://github.com/pps-lab/fl-analysis

*The configuration below is copied from the fl-analysis project, please also credit the original repo. for any reproducing scientific work.*

## Set up the execution

Step 1: Configure your Python (3.7) + tensorflow 2.5.0 environment
```commandline
conda activate <environment_name>
``` 

Step 2: Install the required packages by "pipenv install" and activate it by launching a shell 
```commandline
pipenv shell
```

Step 3: Review the configuration file "config.yml" and run the code by calling command below:
```commandline
python -m src.main -c config.yml
```

## Usage
The configuration of the framework is specified in a config file in YAML format.
A minimal example of a config is shown below.
```yaml
environment:
  num_clients: 3383
  num_selected_clients: 30
  num_malicious_clients: 0
  experiment_name: "Sample run without attackers"

server:
  num_rounds: 80
  num_test_batches: 5
  aggregator:
    name: FedAvg
  global_learning_rate: -1

client:
  clip:
    type: l2
    value: 10
  model_name: resnet18
  benign_training:
    num_epochs: 2
    batch_size: 24
    optimizer: Adam
    learning_rate: 0.001

dataset:
  dataset: femnist
  data_distribution: nonIID
```
The full specification of the supported config options can be found [here](https://pps-lab.com/fl-analysis/)
Some example config files can be find in `train_configs`.


## Output 
Basic training progress is sent to standard output.
More elaborate information is stored in an output folder.
The directory location can be specified through the `XXX` option.
By default, its ... .
The framework stores progress in tfevents, which can be viewed using Tensorboard, e.g.,
```bash
tensorboard --logdir ./experiments/{experiment_name}
```

<!-- LICENSE -->
## License

This project's code is distributed under the MIT License. See `LICENSE` for more information.

