from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import skimage.io
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataset import Dataset as TorchDataset

from ls_axon_segmentation.enums import ClassTag, DataTag
from ls_axon_segmentation.utils import worker_init_function


class LmEnsembleAxonRadiiTestDataSet(TorchDataset):

    PIXEL_SIZE_IN_MICROMETERS = 0.1112

    def __init__(self, file_list, data_directory=None, preprocessing_transform=None):
        self.file_list = file_list
        self.data_directory = data_directory
        self.preprocessing_transform = preprocessing_transform

    def __getitem__(self, idx):
        data_sample_dict = self.file_list[idx]
        input_image_path = Path(data_sample_dict[DataTag.INPUT])
        roi_mask_image_path = Path(data_sample_dict["roi_mask"])
        target_r_arith_path = Path(data_sample_dict["target_r_arith"])
        target_r_eff_path = Path(data_sample_dict["target_r_eff"])
        if self.data_directory is not None:
            input_image_path = Path(self.data_directory).joinpath(input_image_path.as_posix())
            roi_mask_image_path = Path(self.data_directory).joinpath(roi_mask_image_path.as_posix())
            target_r_arith_path = Path(self.data_directory).joinpath(target_r_arith_path.as_posix())
            target_r_eff_path = Path(self.data_directory).joinpath(target_r_eff_path.as_posix())
        input_image = skimage.io.imread(input_image_path.as_posix())
        roi_mask_image = skimage.io.imread(roi_mask_image_path.as_posix()).astype(np.uint8)
        target_r_arith = pd.read_csv(target_r_arith_path.as_posix(), header=None).to_numpy()
        target_r_arith = torch.as_tensor(np.squeeze(target_r_arith))
        target_r_eff = pd.read_csv(target_r_eff_path.as_posix(), header=None).to_numpy()
        target_r_eff = torch.as_tensor(np.squeeze(target_r_eff))

        sample_dict = {
            DataTag.INPUT: input_image,
            "roi_mask": roi_mask_image,
            "target_r_arith": target_r_arith,
            "target_r_eff": target_r_eff,
            "pixel_size_in_micrometers": LmEnsembleAxonRadiiTestDataSet.PIXEL_SIZE_IN_MICROMETERS,
        }

        sample_dict[DataTag.INPUT] = np.moveaxis(sample_dict[DataTag.INPUT], -1, -0)
        sample_dict[DataTag.INPUT] = torch.as_tensor(sample_dict[DataTag.INPUT]).float()
        sample_dict[DataTag.INPUT] /= 255.0
        sample_dict["roi_mask"] = torch.as_tensor(sample_dict["roi_mask"])

        if self.preprocessing_transform is not None:
            sample_dict[DataTag.INPUT] = self.preprocessing_transform(sample_dict[DataTag.INPUT])

        return sample_dict

    def __len__(self):
        return len(self.file_list)


class LmDataSet(TorchDataset):

    CLASS_DICT = {ClassTag.BACKGROUND: 0, "myelin": 1, "axon": 2, ClassTag.IGNORE: 3}

    def __init__(
        self,
        file_list,
        data_directory=None,
        augmentation_transform=None,
        preprocessing_transform=None,
    ):
        self.file_list = file_list
        self.data_directory = data_directory
        self.augmentation_transform = augmentation_transform
        self.preprocessing_transform = preprocessing_transform

    def __getitem__(self, idx):
        data_sample_dict = self.file_list[idx]
        input_image_path = Path(data_sample_dict[DataTag.INPUT])
        mask_image_path = Path(data_sample_dict[DataTag.TARGET])
        if self.data_directory is not None:
            input_image_path = Path(self.data_directory).joinpath(input_image_path.as_posix())
            mask_image_path = Path(self.data_directory).joinpath(mask_image_path.as_posix())
        input = skimage.io.imread(input_image_path.as_posix())
        target = skimage.io.imread(mask_image_path.as_posix()).astype(np.uint8)
        sample_dict = {DataTag.INPUT: input, DataTag.TARGET: target}

        if self.augmentation_transform is not None:
            sample_dict = self.augmentation_transform(sample_dict)

        sample_dict[DataTag.INPUT] = np.moveaxis(sample_dict[DataTag.INPUT], -1, -0)
        sample_dict[DataTag.INPUT] = torch.as_tensor(sample_dict[DataTag.INPUT]).float()
        sample_dict[DataTag.INPUT] /= 255.0
        sample_dict[DataTag.TARGET] = torch.as_tensor(sample_dict[DataTag.TARGET]).long()

        if self.preprocessing_transform is not None:
            sample_dict[DataTag.INPUT] = self.preprocessing_transform(sample_dict[DataTag.INPUT])

        return sample_dict

    def __len__(self):
        return len(self.file_list)


class LmDataModule(pl.LightningDataModule):

    CLASS_DICT = LmDataSet.CLASS_DICT

    def __init__(
        self,
        training_samples,
        validation_samples,
        test_samples_segmentation_metrics,
        test_samples_axon_radii,
        batch_size,
        samples_per_epoch,
        num_workers,
        pin_memory,
        preprocessing_transform=None,
        augmentation_transform=None,
        data_directory=None,
    ):
        super().__init__()
        self.training_data_set = LmDataSet(
            file_list=training_samples,
            data_directory=data_directory,
            augmentation_transform=augmentation_transform,
            preprocessing_transform=preprocessing_transform,
        )
        self.validation_data_set = LmDataSet(
            file_list=validation_samples,
            data_directory=data_directory,
            preprocessing_transform=preprocessing_transform,
        )
        self.test_data_set_segmentation_metrics = LmDataSet(
            file_list=test_samples_segmentation_metrics,
            data_directory=data_directory,
            preprocessing_transform=preprocessing_transform,
        )
        self.test_data_set_axon_radii = LmEnsembleAxonRadiiTestDataSet(
            file_list=test_samples_axon_radii,
            data_directory=data_directory,
            preprocessing_transform=preprocessing_transform,
        )
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @classmethod
    def create_from_file(
        cls,
        training_list_file,
        validation_list_file,
        test_segmentation_metrics_list_file,
        test_axon_radii_samples_list_file,
        *args,
        **kwargs
    ):
        training_samples = OmegaConf.load(training_list_file)
        validation_samples = OmegaConf.load(validation_list_file)
        test_segmentation_metrics_samples = OmegaConf.load(test_segmentation_metrics_list_file)
        test_axon_radii_samples_samples = OmegaConf.load(test_axon_radii_samples_list_file)
        return cls(
            training_samples,
            validation_samples,
            test_segmentation_metrics_samples,
            test_axon_radii_samples_samples,
            *args,
            **kwargs
        )

    def train_dataloader(self):
        if len(self.training_data_set) == 0:
            return None
        training_data_sampler = RandomSampler(
            self.training_data_set, replacement=True, num_samples=self.samples_per_epoch
        )
        return DataLoader(
            dataset=self.training_data_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_function,
            pin_memory=self.pin_memory,
            sampler=training_data_sampler,
        )

    def val_dataloader(self):
        if len(self.validation_data_set) == 0:
            return None
        return DataLoader(
            dataset=self.validation_data_set,
            batch_size=1,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_function,
            pin_memory=self.pin_memory,
        )

    def test_dataloader_segmentation_metrics(self):
        return DataLoader(
            dataset=self.test_data_set_segmentation_metrics,
            batch_size=1,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_function,
            pin_memory=self.pin_memory,
        )

    def test_dataloader_ensemble_mean_axon_radii(self):
        return DataLoader(
            dataset=self.test_data_set_axon_radii,
            batch_size=1,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_function,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return (
            self.test_dataloader_segmentation_metrics(),
            self.test_dataloader_ensemble_mean_axon_radii(),
        )
