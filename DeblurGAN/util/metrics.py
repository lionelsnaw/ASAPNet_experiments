import warnings
import numpy as np

import torch
import torch.nn.functional as F

from math import exp
from typing import Callable, Sequence, Union

from scipy import linalg
from torch import nn
from torch.autograd import Variable

from torchvision.models import inception_v3

from ignite.metrics import PSNR, SSIM
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window


class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations


class FID(Metric):
    r"""Calculates Frechet Inception Distance.

    .. math::
       \text{FID} = |\mu_{1} - \mu_{2}| + \text{Tr}(\sigma_{1} + \sigma_{2} - {2}\sqrt{\sigma_1*\sigma_2})

    where :math:`\mu_1` and :math:`\sigma_1` refer to the mean and covariance of the train data and
    :math:`\mu_2` and :math:`\sigma_2` refer to the mean and covariance of the test data.

    More details can be found in `Heusel et al. 2002`__

    __ https://arxiv.org/pdf/1706.08500.pdf

    In addition, a faster and online computation approach can be found in `Chen et al. 2014`__

    __ https://arxiv.org/pdf/2009.14075.pdf

    Remark:

        This implementation is inspired by pytorch_fid package which can be found `here`__

        __ https://github.com/mseitzer/pytorch-fid

    .. note::
        The default Inception model requires the `torchvision` module to be installed.
        FID also requires `scipy` library for matrix square root calculations.

    Args:
        num_features: number of features predicted by the model or the reduced feature vector of the image.
            Default value is 2048.
        feature_extractor: a torch Module for extracting the features from the input data.
            It returns a tensor of shape (batch_size, num_features).
            If neither ``num_features`` nor ``feature_extractor`` are defined, by default we use an ImageNet
            pretrained Inception Model. If only ``num_features`` is defined but ``feature_extractor`` is not
            defined, ``feature_extractor`` is assigned Identity Function.
            Please note that the model will be implicitly converted to device mentioned in the ``device``
            argument.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            By default, metrics require the output as ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y}``.
        device: specifies which device updates are accumulated on. Setting the
            metric's device to be the same as your ``update`` arguments ensures the ``update`` method is
            non-blocking. By default, CPU.

    Examples:

        .. code-block:: python

            metric = FID()
            metric.attach(default_evaluator, "fid")
            y_true = torch.rand(10, 3, 299, 299)
            y_pred = torch.rand(10, 3, 299, 299)
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["fid"])

        .. testcode::

            metric = FID(num_features=1, feature_extractor=default_model)
            metric.attach(default_evaluator, "fid")
            y_true = torch.ones(10, 4)
            y_pred = torch.ones(10, 4)
            state = default_evaluator.run([[y_pred, y_true]])
            print(state.metrics["fid"])

        .. testoutput::

            0.0
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__(output_transform=output_transform, device=device)
        self.inception_network = PartialInceptionNetwork()
        self.inception_network = self._to_device(self.inception_network)
        self.inception_network.eval()

    def _check_shape_dtype(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output
        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_batchwise_fid = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape_dtype(output)
        y_pred, y = output[0].detach(), output[1].detach()
        
        y_pred = self._to_device(y_pred)
        y = self._to_device(y)
        
        y_pred = (y_pred + 1) / 2   # from [-1, 1] to [0, 1]
        y = (y + 1) / 2             # from [-1, 1] to [0, 1]
        
        y_pred = F.interpolate(y_pred, size=299)
        y = F.interpolate(y, size=299)

        self._sum_of_batchwise_fid += self._calculate_fid(y_pred, y)
        self._num_examples += 1

    @sync_all_reduce("_sum_of_batchwise_fid", "_num_examples")
    def compute(self) -> torch.Tensor:
        if self._num_examples == 0:
            return self._num_examples
        return self._sum_of_batchwise_fid / self._num_examples

    def _to_device(self, elements):
        """
        Transfers elements to cuda if GPU is available
        Args:
            elements: torch.tensor or torch.nn.module
            --
        Returns:
            elements: same as input on GPU memory, if available
        """
        return elements.to(self._device)

    def _get_activations(self, images):
        """
        Calculates activations for last pool layer for all iamges
        --
            Images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
        --
        Returns: np array shape: (N, 2048), dtype: np.float32
        """
        assert images.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                                ", but got {}".format(images.shape)

        num_images = images.shape[0]
        inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
        images = self._to_device(images)
        inception_activations = self.inception_network(images)
        inception_activations = inception_activations.detach().cpu().numpy()
        assert inception_activations.shape == (images.shape[0], 2048), "Expexted output shape to be: {}, but was: {}".format((images.shape[0], 2048), inception_activations.shape)
        return inception_activations

    def _calculate_activation_statistics(self, images):
        """Calculates the statistics used by FID
        Args:
            images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
        Returns:
            mu:     mean over all activations from the last pool layer of the inception model
            sigma:  covariance matrix over all activations from the last pool layer 
                    of the inception model.

        """
        act = self._get_activations(images)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    # Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
                
        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                inception net ( like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                precalcualted on an representive data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def _calculate_fid(self, images1, images2):
        """ Calculate FID between images1 and images2
        Args:
            images1: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
            images2: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
        Returns:
            FID (scalar)
        """
        mu1, sigma1 = self._calculate_activation_statistics(images1)
        mu2, sigma2 = self._calculate_activation_statistics(images2)
        fid = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid
