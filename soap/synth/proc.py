import torch
from torch import Tensor

in1k_mean = torch.tensor([0.485, 0.456, 0.406])
in1k_std = torch.tensor([0.229, 0.224, 0.225])

def in1k_norm(tensor:Tensor, dim:int=-1) -> Tensor:
    '''Normalize a tensor using the ImageNet (in1k) mean and standard deviation.

    Parameters
    ----------
    tensor : Tensor
        The input tensor to be normalized.
    dim : int, optional
        The dimension along which the mean and std are applied. Default is -1.

    Returns
    -------
    Tensor
        The normalized tensor.

    Notes
    -----
    This function adjusts input data (tensor) to have zero mean and unit
    variance according to the ImageNet dataset statistics.
    '''
    shape = [1] * tensor.ndim
    shape[dim] = -1
    mean = in1k_mean.view(shape).to(tensor.device)
    std = in1k_std.reshape(shape).to(tensor.device)
    return (tensor - mean) / std

def in1k_unnorm(tensor:Tensor, dim:int=-1) -> Tensor:
    '''Unnormalize a tensor using the ImageNet (in1k) mean and standard deviation.

    Parameters
    ----------
    tensor : Tensor
        The input tensor to be unnormalized.
    dim : int, optional
        The dimension along which the mean and std are applied. Default is -1.

    Returns
    -------
    Tensor
        The unnormalized tensor.

    Notes
    -----
    This function adjusts input data (tensor) to have the original mean and std
    according to the ImageNet dataset statistics.
    '''
    shape = [1] * tensor.ndim
    shape[dim] = -1
    mean = in1k_mean.view(shape).to(tensor.device)
    std = in1k_std.reshape(shape).to(tensor.device)
    return tensor * std + mean