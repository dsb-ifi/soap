import torch
from torch import Tensor
from .dct import dct2, idct2

def generate_adjusted_pink_noise_dct(
    size: int,
    channels: int,
    batch: int = 1,
    sigma: float = 1.0,
    alpha: float = 1.0,
    cutoff: float | None = None,
    device: torch.device = torch.device('cpu')
) -> Tensor:
    """Generate a 2D pink noise with adjustable frequency decay and optional cutoff in the DCT domain, supporting batch generation.

    Parameters
    ----------
    size: int
        The size of the 2D grid (size x size).
    channels: int
        The number of channels for the output tensor.
    batch: int, optional
        The number of samples to generate in the batch. Default is 1.
    sigma: float
        Scaling factor for the noise amplitude.
    alpha: float
        Exponent for the frequency decay (1/f^alpha).
    cutoff: float, optional
        Cutoff frequency for the decay effect.
    device: torch.device, optional
        The device to use for tensor computations.

    Returns
    -------
    Tensor:
        4D tensor of shape (batch, channels, size, size) representing adjusted pink noise.
    """
    f = torch.arange(size, device=device).float()
    f_x, f_y = torch.meshgrid(f, f, indexing='ij')

    spectrum = torch.sqrt(f_x ** 2 + f_y ** 2)
    spectrum[0, 0] = 1.0  # avoid division by zero at the origin

    if cutoff:
        decay = torch.where(spectrum > cutoff, spectrum ** (-alpha), 1.0)
    else:
        decay = spectrum ** (-alpha)

    # Expand decay to match (batch, channels, size, size) for broadcasting
    decay = decay[None, None, :, :]

    white_noise = torch.randn(batch, channels, size, size, device=device)
    white_noise_dct = dct2(white_noise)
    adjusted_pink_noise_dct = white_noise_dct * decay
    adjusted_pink_noise = idct2(adjusted_pink_noise_dct)
    # Normalize each sample in the batch independently
    adjusted_pink_noise = sigma * adjusted_pink_noise / adjusted_pink_noise.flatten(2).std(dim=2, keepdim=True).unsqueeze(-1)
    return adjusted_pink_noise


