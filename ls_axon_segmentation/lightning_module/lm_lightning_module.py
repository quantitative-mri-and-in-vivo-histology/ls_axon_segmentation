from typing import IO, Callable, Dict, Optional, Union

import hydra.utils
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics

from ls_axon_segmentation.axon_radii_measurement import (
    compute_arithmetic_mean_radius,
    compute_axon_radii,
    compute_effective_radius,
)
from ls_axon_segmentation.enums import ClassTag, DataTag
from ls_axon_segmentation.inference import SlidingWindowInferer
from ls_axon_segmentation.metrics import (
    MaeMultiReferenceError,
    MbeMultiReferenceError,
    RmseMultiReferenceError,
    RsdMultiReferenceError,
    balanced_accuracy,
)
from ls_axon_segmentation.utils import log_image


def flatten_masked(prediction, target, ignore_class=None):
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    if ignore_class is not None:
        mask = target != ignore_class
        target = target[mask]
        prediction = prediction[mask]
    return (prediction, target)


class LmLightningModule(pl.LightningModule):
    def __init__(
        self,
        class_dict,
        net_config,
        loss_config,
        optimizer_config,
        lr_scheduler_config,
        patch_size,
        batch_size,
        use_probabilities_for_loss,
        preprocessing_transform=None,
        log_images_training_every_n_epochs=0,
        log_images_training_batches_per_epoch=0,
        log_images_validation_every_n_epochs=0,
        log_images_validation_batches_per_epoch=0,
        log_images_test_batches_per_epoch=0,
        log_training_loss_name="training_loss",
        log_validation_loss_name="validation_loss",
    ):
        super(LmLightningModule, self).__init__()

        self.save_hyperparameters(
            ignore=[
                "log_images_training_every_n_epochs",
                "log_images_training_batches_per_epoch",
                "log_images_validation_every_n_epochs",
                "log_images_validation_batches_per_epoch",
                "log_images_validation_batches_per_epoch",
                "training_loss_name",
                "validation_loss_name",
            ]
        )

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.net_config = net_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.use_probabilities_for_loss = use_probabilities_for_loss
        self.preprocessing_transform = preprocessing_transform

        self.ignore_class = class_dict.get(ClassTag.IGNORE, None)
        self.class_dict = {
            class_name: class_index for class_name, class_index in class_dict.items() if class_name != ClassTag.IGNORE
        }
        self.number_of_classes = len(self.class_dict)

        self.log_images_training_every_n_epochs = log_images_training_every_n_epochs
        self.log_images_training_batches_per_epoch = log_images_training_batches_per_epoch
        self.log_images_validation_every_n_epochs = log_images_validation_every_n_epochs
        self.log_images_validation_batches_per_epoch = log_images_validation_batches_per_epoch
        self.log_images_test_batches_per_epoch = log_images_test_batches_per_epoch
        self.log_training_loss_name = log_training_loss_name
        self.log_validation_loss_name = log_validation_loss_name

        self.sliding_window_inferer = SlidingWindowInferer(
            roi_size=self.patch_size,
            sw_batch_size=self.batch_size,
            show_progress=False,
            mode="gaussian",
        )

        self.preprocessing_transform = hydra.utils.instantiate(self.preprocessing_transform)
        self.loss_criterion = hydra.utils.instantiate(self.loss_config)
        self.net = hydra.utils.instantiate(self.net_config)
        self.final_activation = torch.nn.Softmax2d()
        self.segmentation_metrics_dict = {
            "accuracy": torchmetrics.functional.accuracy,
            "precision": torchmetrics.functional.precision,
            "recall": torchmetrics.functional.recall,
            "dice": torchmetrics.functional.f1_score,
            "specificity": torchmetrics.functional.specificity,
            "balanced_accuracy": balanced_accuracy,
        }
        self.axon_radii_metrics_dict = {
            "r_eff/normalized_root_mean_square_error": RmseMultiReferenceError(reduce_fn=torch.max, normalize=True),
            "r_eff/normalized_mean_absolute_error": MaeMultiReferenceError(reduce_fn=torch.max, normalize=True),
            "r_eff/normalized_residual_standard_deviation": RsdMultiReferenceError(
                reduce_fn=torch.max, normalize=True, unbiased=False
            ),
            "r_eff/normalized_mean_bias_error": MbeMultiReferenceError(reduce_fn=torch.max, normalize=True),
            "r_arith/normalized_root_mean_square_error": RmseMultiReferenceError(reduce_fn=torch.max, normalize=True),
            "r_arith/normalized_mean_absolute_error": MaeMultiReferenceError(reduce_fn=torch.max, normalize=True),
            "r_arith/normalized_residual_standard_deviation": RsdMultiReferenceError(
                reduce_fn=torch.max, normalize=True, unbiased=False
            ),
            "r_arith/normalized_mean_bias_error": MbeMultiReferenceError(reduce_fn=torch.max, normalize=True),
        }
        for key in self.axon_radii_metrics_dict:
            setattr(self, key, self.axon_radii_metrics_dict[key])

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        state_dict = torch.load(checkpoint_path, map_location)
        net_config = state_dict["hyper_parameters"]["net_config"]
        loss_config = state_dict["hyper_parameters"]["loss_config"]
        lr_scheduler_config = state_dict["hyper_parameters"]["lr_scheduler_config"]
        optimizer_config = state_dict["hyper_parameters"]["optimizer_config"]
        lightning_module = super().load_from_checkpoint(
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            net_config=net_config,
            loss_config=loss_config,
            lr_scheduler_config=lr_scheduler_config,
            optimizer_config=optimizer_config,
            **kwargs,
        )
        return lightning_module

    def configure_optimizers(self):
        self.optimizer = hydra.utils.instantiate(self.optimizer_config, self.net.parameters())
        self.lr_scheduler = hydra.utils.instantiate(self.lr_scheduler_config, self.optimizer)
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}

    def compute_loss(self, logits, target):
        if self.use_probabilities_for_loss:
            probabilities = self.final_activation(logits)
            return self.loss_criterion(probabilities, target)
        else:
            return self.loss_criterion(logits, target)

    def training_step(self, batch, batch_idx):
        image = batch[DataTag.INPUT]
        target = batch[DataTag.TARGET]
        logits = self.net(image)
        loss_val = self.compute_loss(logits, target)

        self.log("lr", self.optimizer.param_groups[0]["lr"])
        self.log("epoch", self.current_epoch)
        self.log(self.log_training_loss_name, loss_val.detach())

        if self._should_log_images(
            batch_idx,
            self.log_images_training_every_n_epochs,
            self.log_images_training_batches_per_epoch,
        ):
            prediction = torch.argmax(logits, 1)
            log_image(self.logger, "training/input", image, self.global_step)
            log_image(self.logger, "training/target", target, self.global_step)
            log_image(self.logger, "training/prediction", prediction, self.global_step)

        return loss_val

    def validation_step(self, batch, batch_idx):
        img = batch[DataTag.INPUT]
        target = batch[DataTag.TARGET]
        logits = self.sliding_window_inferer(img, self.net)
        loss_val = self.compute_loss(logits, target)

        def generate_probabilities(image):
            return self.forward(image, output_type="probabilities")

        out = self.sliding_window_inferer(img, generate_probabilities)
        prediction = torch.argmax(out, 1)

        self.log(self.log_validation_loss_name, loss_val)
        metrics_dict = self.evaluate_segmentation_metrics(prediction, target)
        for metric_name, metric_value in metrics_dict.items():
            self.log("{}_{}".format("validation", metric_name), metric_value)

        if self._should_log_images(
            batch_idx,
            self.trainer.check_val_every_n_epoch * self.log_images_validation_every_n_epochs,
            self.log_images_validation_batches_per_epoch,
        ):
            log_image(self.logger, "validation/{:05d}_input".format(batch_idx), img, self.global_step)
            log_image(self.logger, "validation/{:05d}_target".format(batch_idx), target, self.global_step)
            log_image(
                self.logger,
                "validation/{:05d}_prediction".format(batch_idx),
                prediction,
                self.global_step,
            )

        return loss_val

    def _should_log_images(self, batch_idx, epoch_logging_frequency, batches_per_epoch):

        log_epoch = (
            epoch_logging_frequency == -1
            or self.current_epoch == 0
            or (epoch_logging_frequency > 0 and ((self.current_epoch + 1) % (epoch_logging_frequency) == 0))
        )
        log_batch = batches_per_epoch == -1 or (0 <= batch_idx < batches_per_epoch)

        return log_batch and log_epoch

    def test_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            return self._test_step_segmentation_metrics(batch, batch_idx)
        elif dataloader_idx == 1:
            return self._test_step_ensemble_mean_axon_radii(batch, batch_idx)
        else:
            raise ValueError

    def _test_step_segmentation_metrics(self, batch, batch_idx):
        img = batch[DataTag.INPUT]
        target = batch[DataTag.TARGET]
        prediction = torch.argmax(self.net(img), 1)

        metrics_dict = self.evaluate_segmentation_metrics(prediction, target)
        metrics_dict = {"test_segmentation_metrics_{}".format(k): v for k, v in metrics_dict.items()}
        self.log_dict(metrics_dict, add_dataloader_idx=False)

        if self._should_log_images(batch_idx, -1, self.log_images_test_batches_per_epoch):
            log_image(self.logger, "test/{:05d}_input".format(batch_idx), img, self.global_step)
            log_image(self.logger, "test/{:05d}_target".format(batch_idx), target, self.global_step)
            log_image(
                self.logger,
                "test/{:05d}_prediction".format(batch_idx),
                prediction,
                self.global_step,
            )

        return metrics_dict

    def _test_step_ensemble_mean_axon_radii(self, batch, batch_idx):
        img = batch[DataTag.INPUT]
        roi_mask = batch["roi_mask"]
        target_r_arith = batch["target_r_arith"]
        target_r_eff = batch["target_r_eff"]
        pixel_size_in_micrometers = batch["pixel_size_in_micrometers"]

        def generate_probabilities(image):
            return self.forward(image, output_type="probabilities")

        out = self.sliding_window_inferer(img, generate_probabilities)
        prediction = torch.argmax(out, 1)

        prediction[roi_mask == 0] = self.ignore_class
        prediction_image = prediction.cpu().numpy()
        prediction_image = np.squeeze(prediction_image)

        axon_radii = compute_axon_radii(
            prediction_image == self.class_dict["axon"],
            radius_approximation=["circular_equivalent", "minor_axis"],
            pixel_size=pixel_size_in_micrometers.cpu().numpy(),
        )

        prediction_r_arith = torch.as_tensor(compute_arithmetic_mean_radius(axon_radii["circular_equivalent"])).to(
            self.device
        )
        prediction_r_eff = torch.as_tensor(compute_effective_radius(axon_radii["circular_equivalent"])).to(self.device)
        target_r_arith = torch.squeeze(target_r_arith)
        target_r_eff = torch.squeeze(target_r_eff)

        for metric_name, metric in self.axon_radii_metrics_dict.items():
            if metric_name.startswith("r_arith"):
                metric.update(prediction_r_arith, target_r_arith)
            elif metric_name.startswith("r_eff"):
                metric.update(prediction_r_eff, target_r_eff)
            else:
                raise ValueError
            self.log("test_axon_radii_{}".format(metric_name), metric, add_dataloader_idx=False)

    def evaluate_segmentation_metrics(self, prediction, target):

        present_class_dict = {}
        present_fg_class_dict = {}
        for class_name, class_index in self.class_dict.items():
            if torch.sum(target == class_index) > 0:
                present_class_dict[class_name] = class_index
                if class_name != ClassTag.BACKGROUND:
                    present_fg_class_dict[class_name] = class_index

        prediction_flat, target_flat = flatten_masked(prediction, target, ignore_class=self.ignore_class)
        log_dict = {}

        for metric_name, metric_func in self.segmentation_metrics_dict.items():
            metric_key = "micro_avg/{}".format(metric_name)
            log_dict[metric_key] = metric_func(
                prediction_flat,
                target_flat,
                num_classes=self.number_of_classes,
                average="micro",
            )

            classwise_metric_values = metric_func(
                prediction_flat, target_flat, num_classes=self.number_of_classes, average="none"
            )
            for class_name, class_index in present_class_dict.items():
                metric_key = "{}/{}".format(class_name, metric_name)
                log_dict[metric_key] = classwise_metric_values[class_index]

            if len(present_fg_class_dict) > 0:
                metric_key = "foreground/{}".format(metric_name)
                log_dict[metric_key] = torch.stack(
                    [classwise_metric_values[class_index] for class_name, class_index in present_fg_class_dict.items()]
                ).mean()

        return log_dict

    def forward(self, image, output_type="predictions"):
        self.eval()
        if image.ndim == 3:
            image = torch.unsqueeze(image, 0)
        if image.ndim != 4:
            raise ValueError("Expected 3 (CHW) or 4-dimensional (NCHW) input")
        with torch.no_grad():
            out = self.net(image)
            if output_type == "logits":
                return out
            elif output_type == "probabilities":
                return self.final_activation(out)
            elif output_type == "predictions":
                return torch.unsqueeze(torch.argmax(out, 1), 1)
            else:
                raise ValueError
        return None
