import hydra
from omegaconf import DictConfig, OmegaConf

# add parser for partially instantiated objects
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig) -> None:

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from ls_axon_segmentation import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    if config.get("seed"):
        utils.seed_everything(config.seed, workers=True)
    else:
        utils.seed_everything(None, workers=True)

    lightning_module, data_module, trainer = utils.parse_pl_instances_from_config(config)

    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path is not None:
        checkpoint_path = hydra.utils.to_absolute_path(checkpoint_path)

    trainer.fit(lightning_module, datamodule=data_module, ckpt_path=checkpoint_path)

    if config.get("test_after_training"):
        trainer.test(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    main()
