import torch
from torch import Tensor

from .proc import in1k_norm

def generate_constant(size:int, channels:int, batch:int=1, device:torch.device=torch.device('cpu')) -> Tensor:
    """Generate a constant image tensor.

    Parameters
    ----------
    size: int
        The height and width of the image.
    channels: int
        The number of color channels.
    batch: int
        The number of images to generate.
    value: float
        The constant value to fill the image.

    Returns
    -------
    Tensor:
        A tensor of shape (batch, channels, size, size) filled with the constant value.
    """
    value = in1k_norm(torch.rand(batch, channels,1,1,device=device), 1)
    return torch.zeros(batch, channels, size, size, device=device) + value


def generate_random_gradient(
    size: int, channels: int, batch: int = 1, device: torch.device = torch.device('cpu')
) -> Tensor:
    """
    Generate a batch of images where each image is a linear projection of meshgrid coordinates
    with random RGB coefficients per direction and a bias.
    Returns:
        Tensor of shape (batch, channels, size, size)
    """
    y = torch.linspace(0, 1, steps=size, device=device)
    x = torch.linspace(0, 1, steps=size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy, torch.ones_like(xx)], dim=0)
    coeffs = torch.rand(batch, channels, 3, device=device)
    img = torch.einsum('bck, kxy -> bcxy', coeffs, coords).clip(0,1)
    return in1k_norm(img, 1)


    