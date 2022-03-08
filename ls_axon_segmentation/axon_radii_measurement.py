import numpy as np
from skimage.measure import label, regionprops


def compute_effective_radius(axon_radii):
    axon_radii = np.array(axon_radii)
    return (np.sum(axon_radii ** 6) / np.sum(axon_radii ** 2)) ** (1 / 4)


def compute_arithmetic_mean_radius(axon_radii):
    axon_radii = np.array(axon_radii)
    return np.mean(axon_radii)


def compute_axon_radii(axon_prediction, radius_approximation, pixel_size):

    if not isinstance(radius_approximation, list):
        raise ValueError(
            "radius_approximation must be a list containing one or multiple values of:"
            " 'circular_equivalent', 'minor_axis', 'major axis'."
        )

    axon_labels = label(axon_prediction)
    axon_regionprops = regionprops(axon_labels)
    radii_dict = {}
    scaling_factor = 1 / 2 * pixel_size

    if "circular_equivalent" in radius_approximation:
        radii_dict["circular_equivalent"] = [
            axon_regionprop.equivalent_diameter * scaling_factor for axon_regionprop in axon_regionprops
        ]
    if "minor_axis" in radius_approximation:
        radii_dict["minor_axis"] = [
            axon_regionprop.minor_axis_length * scaling_factor for axon_regionprop in axon_regionprops
        ]
    if "major_axis" in radius_approximation:
        radii_dict["major_axis"] = [
            axon_regionprop.major_axis_length * scaling_factor for axon_regionprop in axon_regionprops
        ]

    return radii_dict
