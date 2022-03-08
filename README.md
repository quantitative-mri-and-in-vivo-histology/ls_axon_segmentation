<div align="center">

# large-scale axon segmentation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](https://zenodo.org/badge/DOI/10.1016/j.neuroimage.2022.118906.svg)](https://doi.org/10.1016/j.neuroimage.2022.118906)
[![Dataset](https://zenodo.org/badge/DOI/10.5281/zenodo.6327740.svg)](https://doi.org/10.5281/zenodo.6327740)
</div>

# Description
Implementation of [Towards a representative reference for MRI-based human axon radius assessment using light microscopy](https://www.sciencedirect.com/science/article/pii/S1053811922000362).

# Installation
## Clone repository
```bash
git clone https://github.com/lau-mo/ls_axon_segmentation
cd ls_axon_segmentation
```
## Requirements
Install the following requirements:
- poetry (find instructions [here](https://python-poetry.org/docs/))
- python >= 3.8
## Install ls_axon_segmentation
To Install ls_axon_segmentation and its dependencies run:
```bash
poetry run pip install setuptools==59.5
poetry install --no-dev
```
## Activate environment
The installation of ls_axon_segmentation using poetry creates a new virtual environment in `/.venv`. Activate the virtual environment, by running
```bash
# Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```
However, if you install ls_axon_segmentation from within an existing environment, no new environment is created in `/.venv`.

# Download data and models
To download training/validation and test data on Linux, run
```bash
wget https://zenodo.org/record/6327740/files/data.zip
unzip data.zip data/*
```
To download the pretrained model on Linux, run
```bash
wget https://zenodo.org/record/6327740/files/models.zip
unzip models.zip models/*
```
Alternatively, you can manually download [data](https://zenodo.org/record/6327740/files/data.zip?download=1) and [pretrained model](https://zenodo.org/record/6327740/files/models.zip?download=1). Then, unzip the contents into `data` and `models` folders, respectively.

# How to run
## Training a model
To train model on one training/validation split, run
```bash
python train.py cv_split=0 +trainer.gpus=1 +num_workers=4
```
To train models on each training/validation split, run
```bash
python train.py -m cv_split=0,1,2,3 +trainer.gpus=1 +num_workers=4
```
To train model on full training/validation dataset and test the model afterwards, run
```bash
python train.py cv_split=full +trainer.gpus=1 +num_workers=4 trainer.min_epochs=200 trainer.max_epochs=200 test_after_training=true
```
The project uses a [template](https://github.com/ashleve/lightning-hydra-template) based on [Hydra](https://hydra.cc/docs/intro/) to dynamically configure the training. See [Store device configuration](#store-device-configuration) on how to store settings for `trainer.gpus` and `num_workers`; see [Customize configuration](#customize-configuration) for details on how to train using different loss function, optimizer, etc.

Logs are written to the `logs` subdirectory. To monitor the training using tensorboard, run
```bash
tensorboard --logdir logs
```
## Testing a model
To test a trained model, run
```bash
python test.py +checkpoint_path=<ckpt_path> +trainer.gpus=1 +num_workers=4
```
E.g. to test the [pretrained model](#download-data-and-models), run
```bash
python test.py +checkpoint_path=models/pretrained_model.ckpt +trainer.gpus=1 +num_workers=4
```

## Predict using trained model
To segment an image using a trained model, run
```bash
python predict.py <model_file> <input_file> <output_file> [--pixel_size] [--extract_extra_measures] [--device]
```
Arguments:
- model_file: model checkpoint
- input_file: input image path
- output_file: output image path (pixel encoding is: 0 -> background, 1 -> myelin, 2 -> axon)

Optional arguments:
- pixel_size: pixel size in micrometers (default: 0.1112)
- extract_extra_measures: whether to extract individual axon radii  and macroscopic measures, i.e., ensemble mean axon radii (arithmetic/effective), volume fractions (axon/myelin/extracellular) and g-ratio. Additional output files are written to <output_file>_individual_axon_radii.csv and <output_file>_macroscopic_measures.csv. Default: no.
- device: device to compute predictions on (options: auto/cpu/cuda; default: auto)

E.g., to predict example image using the pretrained model and extract macroscopic measures, run
```bash
python predict.py models/pretrained_model.ckpt data/examples/cc_isthmus.tiff cc_isthmus_prediction.tiff --extract_extra_measures
```

## Store device configuration
Create a copy of `default.yaml.example` named `default.yaml`. Then, modify default.yaml according to the hardware resources of your machine, i.e. set `trainer.gpus` to 0 (cpu) or 1 (gpu); `num_workers` determines the number of processes used for data preprocessing. E.g.,
```yaml
# @package _global_
trainer:
  gpus: 1
num_workers: 6
```

## Customize configuration
The project uses a [template](https://github.com/ashleve/lightning-hydra-template) based on [Hydra](https://hydra.cc/docs/intro/) to dynamically compose the training configuration, starting with a root configuration in `configs/config.yaml`:
```yaml
# @package _global_

# specify here default training configuration
defaults:
  - _self_
  - trainer: default.yaml
  - logger: many_loggers.yaml # set logger here or use command line (e.g. `python run.py logger=wandb`)
  - callbacks: default.yaml

  # modes are special collections of config options for different purposes, e.g. debugging
  - mode: default.yaml

  # experiment configs allow for version control of specific configurations
  # for example, use them to store best hyperparameters for each combination of model and datamodule
  - experiment: ls_axon_segmentation.yaml

  # optional local config for machine/user specific settings
  - optional local: default.yaml

  # enable color logging
  - override hydra/hydra_logging: colorlog
  - override hydra/job_logging: colorlog
  # modes are special collections of config options for different purposes, e.g. debugging

# path to original working directory
# hydra hijacks working directory by changing it to the new log directory
# so it's useful to have this path as a special variable
# https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
work_dir: ${hydra:runtime.cwd}

# path to folder with data
data_directory: ${work_dir}/data

# pretty print config at the start of the run using Rich library
print_config: True

# disable python warnings if they annoy you
ignore_warnings: True

# evaluate on test set, using best model weights achieved during training
# lightning chooses best weights based on metric specified in checkpoint callback
test_after_training: False

# seed for random number generators in pytorch, numpy and python.random
seed: null

# name of the run, should be used along with experiment mode
name: null

training_loss_name: training_loss
validation_loss_name: validation_loss
validation_metric_name: ${validation_loss_name}

hydra:
  job:
    env_set:
      OMP_NUM_THREADS: "1"
```
The experiment subconfiguration used by default is `experiment/ls_axon_segmentation.yaml`:
```yaml
# @package _global_

defaults:
  - /data_module: lm_data_module.yaml
  - /lightning_module: lm_lightning_module.yaml
  - /loss: lovasz_softmax.yaml
  - /optimizer: sgd.yaml
  - /lr_scheduler: multi_step_lr.yaml
  - /model: u_net_en_encoders.yaml

name: ls_axon_segmentation
batch_size: 4
patch_size: 512
patch_size_2d:
  - ${patch_size}
  - ${patch_size}
```
You can easily create alternative configurations to adjust e.g. loss, optimizer, learining scheduler, batch size and patch size. See `configs/experiment/ls_axon_segmentation_alternative.yaml` for an example:
```yaml
# @package _global_

defaults:
  - /data_module: lm_data_module.yaml
  - /lightning_module: lm_lightning_module.yaml
  - /loss: cross_entropy.yaml
  - /optimizer: adam.yaml
  - /lr_scheduler: step_lr.yaml
  - /model: u_net_en_encoders.yaml

name: ls_axon_segmentation_alt
batch_size: 8
patch_size: 256
patch_size_2d:
  - ${patch_size}
  - ${patch_size}

lightning_module:
  use_probabilities_for_loss: false
```
You can train a model using the alternative configuration by passing the alternative experiment configuration, i.e.,
```bash
python train.py experiment=ls_axon_segmentation_alternative +trainer.gpus=1 +num_workers=4
```
Alternatively, the same can achieved using the command line:
```bash
python train.py loss=cross_entropy optimizer=adam lr_scheduler=step_lr lightning_module.use_probabilities_for_loss=false batch_size=8 patch_size=256 +trainer.gpus=1 +num_workers=4
```
See the [Hydra docs](https://hydra.cc/docs/intro/) for details on configuration and command line interface grammar.

## Generate cross-validation splits
To generate different cross-validation splits, run
```bash
# train on CPU
python generate_cross_validation_splits.py [--k <k_value>] [--seed <seed_value>]
```
Optional arguments:
- k: number of cross validation splits (default: 4)
- seed: random seed used to initialize the random split generator

## Contributors
Laurin Mordhorst, Maria Morozova, Sebastian Papazoglou, Björn Fricke Jan Malte Oeschger, Thibault Tabarin, Henriette Rusch, Carsten Jäger, Stefan Geyer, Nikolaus Weiskopf, Markus Morawski, Siawoosh Mohammadi
<br>
