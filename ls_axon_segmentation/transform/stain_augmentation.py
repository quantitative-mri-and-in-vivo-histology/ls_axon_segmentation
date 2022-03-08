"""
MIT License

Copyright (c) 2018 Peter Byfield

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------------------------------------------------------------------------------

Modifications Copyright (c) 2022 Quantitative MRI and in-vivo histology, UKE Hamburg-Eppendorf
- recomposed below classes into single file
- made number of threads configurable in get_concentrations
"""


import copy
from abc import ABC, abstractmethod

import cv2 as cv
import numpy as np
import spams

from ls_axon_segmentation.enums import DataTag


class StainAugmentation:
    def __init__(self, method, sigma1, sigma2):
        self.method = method
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __call__(self, data):
        input_image = data[DataTag.INPUT]
        stain_augmentor = StainAugmentor(method=self.method, sigma1=self.sigma1, sigma2=self.sigma2)
        stain_augmentor.fit(input_image)
        data[DataTag.INPUT] = stain_augmentor.pop().astype(np.uint8)

        return data


class StainAugmentor(object):
    def __init__(self, method, sigma1=0.2, sigma2=0.2, augment_background=True):
        if method.lower() == "macenko":
            self.extractor = MacenkoStainExtractor
        elif method.lower() == "vahadane":
            self.extractor = VahadaneStainExtractor
        else:
            raise Exception("Method not recognized.")
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.augment_background = augment_background

    def fit(self, I):
        """
        Fit to an image I.
        :param I:
        :return:
        """
        self.image_shape = I.shape
        self.stain_matrix = self.extractor.get_stain_matrix(I)
        self.source_concentrations = get_concentrations(I, self.stain_matrix)
        self.n_stains = self.source_concentrations.shape[1]
        self.tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(I).ravel()

    def pop(self):
        """
        Get an augmented version of the fitted image.
        :return:
        """
        augmented_concentrations = copy.deepcopy(self.source_concentrations)

        for i in range(self.n_stains):
            alpha = np.random.uniform(1 - self.sigma1, 1 + self.sigma1)
            beta = np.random.uniform(-self.sigma2, self.sigma2)
            if self.augment_background:
                augmented_concentrations[:, i] *= alpha
                augmented_concentrations[:, i] += beta
            else:
                augmented_concentrations[self.tissue_mask, i] *= alpha
                augmented_concentrations[self.tissue_mask, i] += beta

        I_augmented = 255 * np.exp(-1 * np.dot(augmented_concentrations, self.stain_matrix))
        I_augmented = I_augmented.reshape(self.image_shape)
        I_augmented = np.clip(I_augmented, 0, 255)

        return I_augmented


class ABCStainExtractor(ABC):
    @staticmethod
    @abstractmethod
    def get_stain_matrix(I):
        """
        Estimate the stain matrix given an image.
        :param I:
        :return:
        """


class MacenkoStainExtractor(ABCStainExtractor):
    @staticmethod
    def get_stain_matrix(I, luminosity_threshold=0.8, angular_percentile=99):
        """
        Stain matrix estimation via method of:
        M. Macenko et al. 'A method for normalizing histology slides for quantitative analysis'

        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param angular_percentile:
        :return:
        """
        assert is_uint8_image(I), "Image should be RGB uint8."
        # Convert to OD and ignore background
        tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(
            I, luminosity_threshold=luminosity_threshold
        ).reshape((-1,))
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        OD = OD[tissue_mask]

        # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
        _, V = np.linalg.eigh(np.cov(OD, rowvar=False))

        # The two principle eigenvectors
        V = V[:, [2, 1]]

        # Make sure vectors are pointing the right way
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1

        # Project on this basis.
        That = np.dot(OD, V)

        # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
        phi = np.arctan2(That[:, 1], That[:, 0])

        # Min and max angles
        minPhi = np.percentile(phi, 100 - angular_percentile)
        maxPhi = np.percentile(phi, angular_percentile)

        # the two principle colors
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

        # Order of H and E.
        # H first row.
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])

        return normalize_matrix_rows(HE)


class VahadaneStainExtractor(ABCStainExtractor):
    @staticmethod
    def get_stain_matrix(I, luminosity_threshold=0.8, regularizer=0.1):
        """
        Stain matrix estimation via method of:
        A. Vahadane et al. 'Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images'
        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param regularizer:
        :return:
        """
        assert is_uint8_image(I), "Image should be RGB uint8."
        # convert to OD and ignore background
        tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(
            I, luminosity_threshold=luminosity_threshold
        ).reshape((-1,))
        OD = convert_RGB_to_OD(I).reshape((-1, 3))
        OD = OD[tissue_mask]

        # do the dictionary learning
        dictionary = spams.trainDL(
            X=OD.T, K=2, lambda1=regularizer, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False
        ).T

        # order H and E.
        # H on first row.
        if dictionary[0, 0] < dictionary[1, 0]:
            dictionary = dictionary[[1, 0], :]

        return normalize_matrix_rows(dictionary)


class LuminosityThresholdTissueLocator:
    @staticmethod
    def get_tissue_mask(I, luminosity_threshold=0.8):
        """
        Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
        Typically we use to identify tissue in the image and exclude the bright white background.
        :param I: RGB uint 8 image.
        :param luminosity_threshold: Luminosity threshold.
        :return: Binary mask.
        """
        assert is_uint8_image(I), "Image should be RGB uint8."
        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
        mask = L < luminosity_threshold

        # Check it's not empty
        if mask.sum() == 0:
            raise TissueMaskException("Empty tissue mask computed")

        return mask


def convert_RGB_to_OD(I):
    """
    Convert from RGB to optical density (OD_RGB) space.
    RGB = 255 * exp(-1*OD_RGB).
    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    mask = I == 0
    I[mask] = 1
    return np.maximum(-1 * np.log(I / 255), 1e-6)


def convert_OD_to_RGB(OD):
    """
    Convert from optical density (OD_RGB) to RGB.
    RGB = 255 * exp(-1*OD_RGB)
    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, "Negative optical density."
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def get_concentrations(I, stain_matrix, regularizer=0.01, num_threads=1):
    """
    Estimate concentration matrix given an image and stain matrix.
    :param I:
    :param stain_matrix:
    :param regularizer:
    :return:
    """
    OD = convert_RGB_to_OD(I).reshape((-1, 3))
    return (
        spams.lasso(X=OD.T, D=stain_matrix.T, mode=2, lambda1=regularizer, pos=True, numThreads=num_threads).toarray().T
    )


def get_sign(x):
    """
    Returns the sign of x.
    :param x: A scalar x.
    :return: The sign of x.
    """

    if x > 0:
        return +1
    elif x < 0:
        return -1
    elif x == 0:
        return 0


def normalize_matrix_rows(A):
    """
    Normalize the rows of an array.
    :param A: An array.
    :return: Array with rows normalized.
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


def is_image(I):
    """
    Is I an image.
    """
    if not isinstance(I, np.ndarray):
        return False
    if not I.ndim == 3:
        return False
    return True


def is_uint8_image(I):
    """
    Is I a uint8 image.
    """
    if not is_image(I):
        return False
    if I.dtype != np.uint8:
        return False
    return True


class TissueMaskException(Exception):
    pass
