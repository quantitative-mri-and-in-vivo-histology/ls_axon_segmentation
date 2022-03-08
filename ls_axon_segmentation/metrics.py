import torch
import torchmetrics
from torchmetrics import Metric


def mean_absolute_error(prediction, target, normalize=False):
    mae = torch.mean(torch.abs(prediction - target))
    if normalize:
        mae /= torch.mean(torch.abs(target))
    return mae


def root_mean_squared_error(prediction, target, normalize=False):
    rmse = torch.sqrt(torch.mean((prediction - target) ** 2))
    if normalize:
        rmse /= torch.mean(target)
    return rmse


def mean_bias_error(prediction, target, normalize=False):
    mbe = torch.mean(prediction - target)
    if normalize:
        mbe /= torch.mean(target)
    return mbe


def residual_standard_deviation(prediction, target, normalize=False, unbiased=True):
    rsd = torch.std(prediction - target, unbiased)
    if normalize:
        rsd /= torch.mean(target)
    return rsd


def balanced_accuracy(prediction, target, *args, **kwargs):
    return (
        torchmetrics.functional.specificity(prediction, target, *args, **kwargs)
        + torchmetrics.functional.recall(prediction, target, *args, **kwargs)
    ) / 2.0


class MultiReferenceErrorMetric(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.predictions.append(preds)
        self.targets.append(target)

    def compute(self):
        predictions = torch.stack(self.predictions)
        targets = torch.stack(self.targets)

        print(predictions)
        print(targets)

        if targets.ndim == 1:
            return self.compute_one(predictions, targets)
        if targets.ndim == 2:
            errors = torch.stack([self.compute_one(predictions, targets[:, c]) for c in range(targets.shape[1])])
            return self.reduce_fn(errors)


class RmseMultiReferenceError(MultiReferenceErrorMetric):
    def __init__(self, dist_sync_on_step=False, reduce_fn=torch.max, normalize=False, unbiased=True):
        super(RmseMultiReferenceError, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_fn = reduce_fn
        self.normalize = normalize
        self.unbiased = unbiased

    def compute_one(self, predictions, targets):
        return root_mean_squared_error(predictions, targets, normalize=self.normalize)


class RsdMultiReferenceError(MultiReferenceErrorMetric):
    def __init__(self, dist_sync_on_step=False, reduce_fn=torch.max, normalize=False, unbiased=False):
        super(RsdMultiReferenceError, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_fn = reduce_fn
        self.normalize = normalize
        self.unbiased = unbiased

    def compute_one(self, predictions, targets):
        return residual_standard_deviation(predictions, targets, normalize=self.normalize, unbiased=self.unbiased)


class MbeMultiReferenceError(MultiReferenceErrorMetric):
    def __init__(self, dist_sync_on_step=False, reduce_fn=torch.max, normalize=False, unbiased=True):
        super(MbeMultiReferenceError, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_fn = reduce_fn
        self.normalize = normalize
        self.unbiased = unbiased

    def compute_one(self, predictions, targets):
        return mean_bias_error(predictions, targets, normalize=self.normalize)


class MaeMultiReferenceError(MultiReferenceErrorMetric):
    def __init__(self, dist_sync_on_step=False, reduce_fn=torch.max, normalize=False):
        super(MaeMultiReferenceError, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self.reduce_fn = reduce_fn
        self.normalize = normalize

    def compute_one(self, predictions, targets):
        return mean_absolute_error(predictions, targets, normalize=self.normalize)
