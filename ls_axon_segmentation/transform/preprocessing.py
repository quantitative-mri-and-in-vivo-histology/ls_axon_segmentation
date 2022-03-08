import torchvision.transforms.functional as tvf


class StandardizeChannelwise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return tvf.normalize(data, self.mean, self.std, inplace=False)
