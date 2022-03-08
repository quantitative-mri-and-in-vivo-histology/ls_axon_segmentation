import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
import torch
from tifffile import imwrite

from ls_axon_segmentation.axon_radii_measurement import (
    compute_arithmetic_mean_radius,
    compute_axon_radii,
    compute_effective_radius,
)
from ls_axon_segmentation.inference import sliding_window_inference
from ls_axon_segmentation.lightning_module.lm_lightning_module import LmLightningModule
from ls_axon_segmentation.utils import get_logger


log = get_logger(__name__)


def compute_ppi_from_pixel_size(pixel_size_in_micrometer):
    return 2.54 / (pixel_size_in_micrometer * 1e-4)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Segment light microscopy sections.")
    parser.add_argument("model_file", type=str, help="Model to use for prediction.")
    parser.add_argument("input_file", type=str, help="Input image to predict.")
    parser.add_argument("output_file", type=str, help="Path to store predicted image.")
    parser.add_argument(
        "--extract_extra_measures",
        default=False,
        action="store_true",
        help="Whether to extract axon radii and macroscopic measures.",
    )
    parser.add_argument("--pixel_size", default=0.1112, type=float, help="Pixel size in micrometers.")
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device to use for prediction. Options: cpu, cuda.",
    )
    args = parser.parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    log.info(f"Using device: {device}.")

    log.info(f"Loading model from {args.model_file}.")
    model = LmLightningModule.load_from_checkpoint(args.model_file, map_location=device)
    model.to(device)
    log.info("... done loading model.")

    log.info(f"Loading input image from {args.input_file}.")
    input_image = skimage.io.imread(args.input_file) / 255.0
    log.info("... done loading input image.")

    with torch.no_grad():
        log.info("Predicting ...")
        input_image = np.moveaxis(input_image, -1, 0)
        input_tensor = torch.as_tensor(input_image).float()
        input_tensor = torch.unsqueeze(input_tensor, 0)
        input_tensor = model.preprocessing_transform(input_tensor)

        def generate_probabilities(image):
            return model(image, output_type="probabilities")

        prediction_tensor = sliding_window_inference(
            input_tensor,
            predictor=generate_probabilities,
            roi_size=model.patch_size,
            sw_batch_size=model.batch_size,
            mode="gaussian",
            show_progress=True,
            sw_device=device,
            device="cpu",
        )
        prediction_tensor = torch.argmax(prediction_tensor, 1)
        prediction_image = prediction_tensor.numpy().astype(np.uint8)
        prediction_image = np.squeeze(prediction_image)
        log.info("... done predicting.")

    resolution_in_ppi = [compute_ppi_from_pixel_size(args.pixel_size) for _ in range(2)]

    log.info(f"Saving segmentation to {args.output_file}...")
    imwrite(
        args.output_file,
        prediction_image,
        {"resolution": (*resolution_in_ppi, "INCH"), "compress": 5},
    )
    log.info(f"... done saving segmentation to {args.output_file}.")

    if args.extract_extra_measures:
        log.info("Extracting individual axon radii...")
        individual_axon_radii_dict = compute_axon_radii(
            prediction_image == 2,
            ["minor_axis", "major_axis", "circular_equivalent"],
            args.pixel_size,
        )
        log.info("... done extracting individual axon radii.")

        individual_axon_radii_file = Path(args.output_file).parent.joinpath(
            "{}_individual_axon_radii.csv".format(Path(args.output_file).stem)
        )
        log.info(f"Saving individual axon radii to {individual_axon_radii_file.as_posix()}")
        pd.DataFrame.from_dict(individual_axon_radii_dict).to_csv(individual_axon_radii_file.as_posix(), index=False)
        log.info(f"... done saving individual axon radii to {individual_axon_radii_file.as_posix()}.")

        log.info("Computing macroscopic measures ...")
        macroscopic_measures_dict = {}
        for radius_approximation, individual_axon_radii in individual_axon_radii_dict.items():
            macroscopic_measures_dict["r_eff_{}".format(radius_approximation)] = [
                compute_effective_radius(individual_axon_radii)
            ]
            macroscopic_measures_dict["r_arith_{}".format(radius_approximation)] = [
                compute_arithmetic_mean_radius(individual_axon_radii)
            ]
        log.info("... done computing ensemble mean axon radii.")

        total_size = np.size(prediction_image)
        macroscopic_measures_dict["avf"] = np.count_nonzero(prediction_image == 2) / total_size
        macroscopic_measures_dict["mvf"] = np.count_nonzero(prediction_image == 1) / total_size
        macroscopic_measures_dict["evf"] = np.count_nonzero(prediction_image == 0) / total_size
        macroscopic_measures_dict["g_ratio"] = np.nan
        if macroscopic_measures_dict["mvf"] > 0 and macroscopic_measures_dict["avf"] > 0:
            macroscopic_measures_dict["g_ratio"] = np.sqrt(
                1
                - (
                    macroscopic_measures_dict["mvf"]
                    / (macroscopic_measures_dict["avf"] + macroscopic_measures_dict["mvf"])
                )
            )

        macroscopics_measures_file = Path(args.output_file).parent.joinpath(
            "{}_macroscopic_measures.csv".format(Path(args.output_file).stem)
        )
        log.info(f"Saving macroscopic measures to {macroscopics_measures_file.as_posix()}")
        pd.DataFrame.from_dict(macroscopic_measures_dict).to_csv(macroscopics_measures_file.as_posix(), index=False)
        log.info(f"... done saving macroscopic measures axon radii to" f" {macroscopics_measures_file.as_posix()}.")
