from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from ls_axon_segmentation.enums import DataTag


class ImgAugAugmentation:
    def __init__(self, augmentation_sequence):
        self._augmentation_sequence = augmentation_sequence

    def __call__(self, data):
        input = data[DataTag.INPUT]
        target = data[DataTag.TARGET]
        segmap = SegmentationMapsOnImage(target, shape=input.shape)
        augmented_input, augmented_target = self._augmentation_sequence(image=input, segmentation_maps=segmap)
        augmented_target = augmented_target.get_arr()
        data[DataTag.INPUT] = augmented_input
        data[DataTag.TARGET] = augmented_target

        return data
