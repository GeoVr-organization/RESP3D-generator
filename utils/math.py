import torch
from torch import Tensor


def minmax(x: Tensor) -> Tensor:
    r"""
    Rescale linearly the input's values between 0 and 1:

    .. math:: \mathit{x[i]} =
        \frac{x[i] - \min(x)}{\max(\max(x) - \min(x), \varepsilon)}

    :param x: The input tensor.
    :type x: Tensor
    :return: The rescaled version between 0 and 1.
    :rtype: Tensor
    """
    xmin, xmax = torch.aminmax(x)
    return (x - xmin) / max(xmax - xmin, torch.finfo(x.dtype).eps)
