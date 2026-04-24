import torch
from torch import Tensor

from .proc import in1k_norm, in1k_unnorm
from .pinknoise import generate_adjusted_pink_noise_dct
from .gradient import generate_constant, generate_random_gradient


def pink_modulated_white(size:int, channels:int, batch:int=1, device:torch.device=torch.device('cpu')) -> Tensor:
    """Generate a modulated white noise tensor.

    Parameters
    ----------
    size : int
        The height and width of the images.
    channels : int
        The number of color channels in the images.
    batch : int
        The number of images to generate.
    device : torch.device
        The device to run the generation on.
    """
    mod = generate_adjusted_pink_noise_dct(size, 1, batch, device=device).expand(batch, channels, size, size)
    white = torch.randn(batch, channels, size, size, device=device)
    return in1k_unnorm(mod,1).clip(0,1) * white


def synthesize_marginal(
    size:int, channels:int, batch:int, device:torch.device, weights:list[float]=[.05,.15,.4,.3,.1]
) -> Tensor:
    '''Generate a batch of non-informative synthesized with varying noise patterns.

    Parameters
    ----------
    size : int
        The height and width of the images.
    channels : int
        The number of color channels in the images.
    batch : int
        The number of images to generate.
    device : torch.device
        The device to run the generation on.

    Returns
    -------
    Tensor:
        A batch of synthesized images.
    '''
    # Random choice of uniform, gaussian, pink noise, or gradient / constant image
    # 0: uniform (normalized with in1k_norm)
    # 1: gaussian (by default; normalized)
    # 2: pink noise (already normalized)
    # 3: gradient (already normalized)
    # 4: constant (already normalized)

    w = torch.tensor(weights, device=device)
    num_samples = torch.multinomial(w, batch, replacement=True)
    output = []
    for i in range(5):
        idx = (num_samples == i).nonzero(as_tuple=True)[0]
        n = idx.numel()
        if n == 0:
            continue
        if i == 0:
            output.append(in1k_norm(torch.rand(n, channels, size, size, device=device), 1))
        elif i == 1:
            output.append(torch.randn(n, channels, size, size, device=device))
        elif i == 2:
            output.append(generate_adjusted_pink_noise_dct(size, channels, n, device=device))
        elif i == 3:
            output.append(generate_random_gradient(size, channels, n, device=device))
        elif i == 4:
            output.append(generate_constant(size, channels, n, device=device))

    # Concatenate and shuffle
    out = torch.cat(output, dim=0)
    perm = torch.randperm(batch, device=device)
    return out[perm]


def synthesize_mixture(
    size:int, channels:int, batch:int, device:torch.device, alpha:list[float]=[1.0, 1.0, 2.0]
) -> Tensor:
    """Generate a batch of non-informative synthesized images with mixed noise patterns.

    Mixes the following synthesized images:
        - Modulated white noise
        - Pink Noise
        - Gradient

    Parameters
    ----------
    size : int
        The height and width of the images.
    channels : int
        The number of color channels in the images.
    batch : int
        The number of images to generate.
    device : torch.device
        The device to run the generation on.
    weights : list[float]
        Weights for each noise type (3 types).

    Returns
    -------
    Tensor: 
        A batch of synthesized images.
    """
    a = torch.tensor(alpha, device=device)
    out = torch.zeros(batch, channels, size, size, device=device)
    weights = torch.distributions.Dirichlet(a).sample((batch,)).view(batch, 3, 1, 1, 1)
    out += weights[:,0] * pink_modulated_white(size, channels, batch, device=device)
    out += weights[:,1] * generate_adjusted_pink_noise_dct(size, channels, batch, device=device)
    out += weights[:,2] * generate_random_gradient(size, channels, batch, device=device)
    return out
    
