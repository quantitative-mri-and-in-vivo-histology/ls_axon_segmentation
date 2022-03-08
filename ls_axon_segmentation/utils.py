import logging
import math
import random
import warnings
from typing import List, Sequence

import hydra
import imgaug
import numpy as np
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.seed import pl_worker_init_function

log = logging.getLogger(__name__)


def parse_pl_instances_from_config(config):
    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.data_module._target_}>")
    data_module: pl.LightningDataModule = hydra.utils.instantiate(config.data_module)

    # Init lightning model
    log.info(f"Instantiating model <{config.lightning_module._target_}>")
    lightning_module: pl.LightningModule = hydra.utils.instantiate(config.lightning_module, _recursive_=False)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    log_hyperparameters(
        config=config,
        lightning_module=lightning_module,
        data_module=data_module,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    return lightning_module, data_module, trainer


def seed_everything(seed, workers=True):
    seed = pl.seed_everything(seed, workers)
    imgaug.random.seed(seed)
    return seed


def worker_init_function(worker_id, rank=None):
    pl_worker_init_function(worker_id, rank)
    global_rank = rank if rank is not None else rank_zero_only.rank
    process_seed = torch.initial_seed()
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    imgaug.random.seed(ss.generate_state(4))


def get_tb_logger(logger):
    if isinstance(logger, pl.loggers.base.LoggerCollection):
        loggers_to_check = logger
    else:
        loggers_to_check = [logger]

    for potential_tb_logger in loggers_to_check:
        if isinstance(potential_tb_logger, pl.loggers.tensorboard.TensorBoardLogger):
            return potential_tb_logger
    return None


def log_image(logger, name, image, step):
    tb_logger = get_tb_logger(logger)

    if tb_logger is None:
        return

    image = image.float()
    if image.ndim == 2:
        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image, 0)
    elif image.ndim == 3:
        image = torch.unsqueeze(image, 1)

    number_of_samples = 9
    if number_of_samples is not None:
        number_of_samples = min(image.shape[0], number_of_samples)
        image = image[0:number_of_samples, ...]
    grid_rows = math.ceil(math.sqrt(image.shape[0]))
    grid = torchvision.utils.make_grid(image, nrow=grid_rows, normalize=True)

    tb_logger.experiment.add_image(name, grid, step)


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # if config.trainer.get("gpus"):
        #     config.trainer.gpus = 0
        # if config.data_module.get("pin_memory"):
        #     config.data_module.pin_memory = False
        # if config.data_module.get("num_workers"):
        #     config.data_module.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = list(config.keys())

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    lightning_module: pl.LightningModule,
    data_module: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of model parameters
    """
    hparams = {}
    hparams.update(config)

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in lightning_module.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in lightning_module.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in lightning_module.parameters() if not p.requires_grad)

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
