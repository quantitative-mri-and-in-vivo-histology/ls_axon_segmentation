import argparse
import logging
from pathlib import Path

from omegaconf import OmegaConf
from sklearn.model_selection import KFold
from ls_axon_segmentation.utils import get_logger


log = get_logger(__name__)


def collect_samples(data_set_directory_path):

    image_paths = sorted(data_set_directory_path.glob("training_validation/*_image.tiff"))
    mask_paths = sorted(data_set_directory_path.glob("training_validation/*_mask.tiff"))
    assert len(image_paths) == len(mask_paths), "Number of inputs and masks is not equal."

    samples = []
    for (image_path, mask_path) in zip(image_paths, mask_paths):
        samples.append(
            {
                "input": image_path.relative_to(data_set_directory_path).as_posix(),
                "target": mask_path.relative_to(data_set_directory_path).as_posix(),
            }
        )

    return samples


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Generate training/validation split for k-fold crossvalidation.")
    parser.add_argument("--k", type=int, default=4, help="Number of folds for cross validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    number_of_splits = args.k
    random_state = args.seed
    data_set_directory_path = Path(".").joinpath("data")
    output_directory_path = data_set_directory_path
    samples = collect_samples(data_set_directory_path)

    log.info(f"Generating split of data set into {number_of_splits} splits.")
    kf = KFold(n_splits=number_of_splits, shuffle=True, random_state=random_state)

    cv_names = ["{}".format(i) for i in range(number_of_splits)]
    cv_names.append("full")

    splits = list(kf.split(samples))
    full_training_indices = list(range(len(samples)))
    full_validation_indices = []
    splits.append((full_training_indices, full_validation_indices))

    for split_name, (training_indices, validation_indices) in zip(cv_names, splits):

        log.info(f"Preparing {split_name}.")

        training_samples = [samples[training_index] for training_index in training_indices]
        validation_samples = [samples[index] for index in validation_indices]

        log.info(f"Saving training file list for cv split {split_name}.")
        training_samples_config_path = output_directory_path.joinpath("training_cv_{}.yaml".format(split_name))
        OmegaConf.save(OmegaConf.create(training_samples), training_samples_config_path.as_posix())
        log.info(f"Saved training file list for {split_name}" f" to {training_samples_config_path.as_posix()}")

        log.info(f"Saving validation file list for cv split {split_name}.")
        validation_samples_config_path = output_directory_path.joinpath("validation_cv_{}.yaml".format(split_name))
        OmegaConf.save(OmegaConf.create(validation_samples), validation_samples_config_path.as_posix())
        log.info(f"Saved training file list for {split_name}" f" to {validation_samples_config_path.as_posix()}")
