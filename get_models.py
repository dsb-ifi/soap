import timm
import torch
import sys, os
from types import MethodType
import torch.nn as nn
import math
import numbers
from torch import Size, Tensor
from torch.nn import functional as F, init
from torch.nn.parameter import Parameter
from typing import Optional, Union, Literal
from jaxtyping import Float, Int

def get_patch_size(model:str):
    match model:
        case 'dino_base': return 16
        case 'dinov2_base': return 14
        case 'mae_base' : return 16
        case 'deit3_base': return 16
        case 'capi_large': return 14

def get_dino_base():
    model = timm.create_model('vit_base_patch16_224.dino', pretrained=True, dynamic_img_size=True)
    patch_size = get_patch_size('dino_base')
    num_global_tokens = 1
    patch_indices = None
    return model, patch_size, num_global_tokens, patch_indices

def get_dinov2_base():
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', True, dynamic_img_size=True)
    patch_size = get_patch_size('dinov2_base')
    num_global_tokens = 5
    patch_indices = None
    return model, patch_size, num_global_tokens, patch_indices

def get_mae_base():
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=True, dynamic_img_size=True)
    patch_size = get_patch_size('mae_base')
    num_global_tokens = 1
    patch_indices = None
    return model, patch_size, num_global_tokens, patch_indices

def get_deit3_base():
    model = timm.create_model('deit3_base_patch16_224.fb_in1k', pretrained=True, dynamic_img_size=True)
    patch_size = get_patch_size('deit3_base')
    num_global_tokens = 1
    patch_indices = None
    return model, patch_size, num_global_tokens, patch_indices

def get_capi_large():
    model = torch.hub.load('facebookresearch/capi:main', 'capi_vitl14_in22k')
    patch_size = get_patch_size('capi_large')
    num_global_tokens = model.num_prefix_tokens
    patch_indices = None
    return model, patch_size, num_global_tokens, patch_indices


def get_dense_backbone(model:str):
    if model == 'dino_base':
        model, patch_size, num_global_tokens, patch_indices = get_dino_base()
        forward_fn = dinov2_forward # Same as DINOv2
    elif model == "dinov2_base":
        model, patch_size, num_global_tokens, patch_indices = get_dinov2_base()
        forward_fn = dinov2_forward
    elif model == "capi_large":
        model, patch_size, num_global_tokens, patch_indices = get_capi_large()
        forward_fn = capi_forward
    elif model == "mae_base":
        model, patch_size, num_global_tokens, patch_indices = get_mae_base()
        forward_fn = dinov2_forward # Same as DINOv2
    elif model == 'deit3_base':
        model, patch_size, num_global_tokens, patch_indices = get_deit3_base()
        forward_fn = dinov2_forward # Same as DINOv2
    else:
        msg = f'Backbone {model} has not been set up for salient segmentation.'
        raise NotImplementedError(msg)

    model.patch_size = patch_size
    model.feat_dim = model.embed_dim # For consistency
    model.forward = MethodType(forward_fn, model)
    return model

def dinov2_forward(
    self,
    x: Float[Tensor, "b c h w"],
    vit_feat: Literal["k", "q", "v", "kqv", "out"] = "out",
    vit_layer: int=-1
) -> tuple[Float[Tensor, "b d"], Float[Tensor, "b k d"], Float[Tensor, "b ih iw d"]]:
    """Used at eval time"""
    bs, _, h, w = x.shape

    msg = f'Got vit_layer index {vit_layer}, but dino_base only has {len(self.blocks)} blocks.'
    assert -(len(self.blocks)+1) < vit_layer < len(self.blocks), msg
    out_block = vit_layer % len(self.blocks)

    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output

    def hook_fn_forward_out(module, input, output):
        feat_out["out"] = output
    
    if vit_feat != "out":
        self.blocks[out_block].attn.qkv.register_forward_hook(hook_fn_forward_qkv)
    elif out_block < len(self.blocks) - 1:
        last_module = list(self.blocks[out_block].children())[-1]
        handle = last_module.register_forward_hook(hook_fn_forward_out)

    with torch.no_grad() :
        h, w = x.shape[2], x.shape[3]
        feat_h, feat_w = h // self.patch_size, w // self.patch_size
        out = self.forward_features(x)
        bs, nb_token, c = out.shape
        out_shape = (bs, feat_h * feat_w, self.feat_dim)
        if vit_feat == "out":
            if out_block < len(self.blocks) - 1:
                out = feat_out["out"]
            return out[:,self.num_prefix_tokens:].reshape(*out_shape)

        qkv = (
                feat_out["qkv"]
                .reshape(bs, nb_token, 3, c)
                .permute(2, 0, 3, 1)
            )
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k.transpose(1, 2).reshape(bs, nb_token, -1)
        q = q.transpose(1, 2).reshape(bs, nb_token, -1)
        v = v.transpose(1, 2).reshape(bs, nb_token, -1)

        if vit_feat == "k":
            feats = k[:, self.num_prefix_tokens:].reshape(*out_shape)
        elif vit_feat == "q":
            feats = q[:, self.num_prefix_tokens:].reshape(*out_shape)
        elif vit_feat == "v":
            feats = v[:, self.num_prefix_tokens:].reshape(*out_shape)
        elif vit_feat == "kqv":
            k = k[:, self.num_prefix_tokens:].reshape(*out_shape)
            q = q[:, self.num_prefix_tokens:].reshape(*out_shape)
            v = v[:, self.num_prefix_tokens:].reshape(*out_shape)
            feats = torch.cat([k, q, v], dim=-1)

        return feats


def capi_forward(
    self,
    x: Float[Tensor, "b c h w"],
    vit_feat: Literal["k", "q", "v", "kqv", "out"] = "out"
    ):
    b, _, h, w = x.shape

    # Hooks to get the k, q, v
    feat_out = {}
    def hook_fn_forward(module, input, output):
        feat_out["x"] = output

    def hook_fn_forward_k(module, input, output):
        feat_out["k"] = output

    def hook_fn_forward_q(module, input, output):
        feat_out["q"] = output

    def hook_fn_forward_v(module, input, output):
        feat_out["v"] = output
        
    if vit_feat == "k":
        self.encoder.blocks[-1].residual1.fn.k_proj.register_forward_hook(hook_fn_forward)
    elif vit_feat == "q":
        self.encoder.blocks[-1].residual1.fn.q_proj.register_forward_hook(hook_fn_forward)
    elif vit_feat == "v":
        self.encoder.blocks[-1].residual1.fn.v_proj.register_forward_hook(hook_fn_forward)
    elif vit_feat == "kqv":
        self.encoder.blocks[-1].residual1.fn.k_proj.register_forward_hook(hook_fn_forward_k)
        self.encoder.blocks[-1].residual1.fn.q_proj.register_forward_hook(hook_fn_forward_q)
        self.encoder.blocks[-1].residual1.fn.v_proj.register_forward_hook(hook_fn_forward_v)

    with torch.no_grad() :
        enc_out, _ = self.forward_features(x, None, None, enc_layer=self.out_layer, dec_layer=None)
        if vit_feat == "out":
            return enc_out[:, self.num_prefix_tokens:]
        elif vit_feat == "kqv":
            k, q, v = (feat_out[out][:, self.num_prefix_tokens:] for out in ["k", "q", "v"])
            return torch.cat([k, q, v], dim=-1)
        else:
            return feat_out["x"][:, self.num_prefix_tokens:]



##################################
### RMSNorm is needed for CAPI ###
##################################

_shape_t = Union[int, list[int], Size]

def rms_norm(x, normalized_shape, weight=None, eps=None):
    dims = tuple(range(-len(normalized_shape), 0))
    rms = torch.sqrt(x.pow(2).mean(dim=dims, keepdim=True) + (eps or torch.finfo(x.dtype).eps))
    out = x / rms
    if weight is not None:
        out = out * weight
    return out

if not hasattr(nn, "RMSNorm"):
    class RMSNorm(nn.Module):
        r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

        This layer implements the operation as described in
        the paper `Root Mean Square Layer Normalization <https://arxiv.org/pdf/1910.07467.pdf>`__

        .. math::
            y_i = \frac{x_i}{\mathrm{RMS}(x)} * \gamma_i, \quad
            \text{where} \quad \text{RMS}(x) = \sqrt{\epsilon + \frac{1}{n} \sum_{i=1}^{n} x_i^2}

        The RMS is taken over the last ``D`` dimensions, where ``D``
        is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
        is ``(3, 5)`` (a 2-dimensional shape), the RMS is computed over
        the last 2 dimensions of the input.

        Args:
            normalized_shape (int or list or torch.Size): input shape from an expected input
                of size

                .. math::
                    [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                        \times \ldots \times \text{normalized\_shape}[-1]]

                If a single integer is used, it is treated as a singleton list, and this module will
                normalize over the last dimension which is expected to be of that specific size.
            eps: a value added to the denominator for numerical stability. Default: ``torch.finfo(x.dtype).eps``
            elementwise_affine: a boolean value that when set to ``True``, this module
                has learnable per-element affine parameters initialized to ones (for weights). Default: ``True``.

        Shape:
            - Input: :math:`(N, *)`
            - Output: :math:`(N, *)` (same shape as input)

        Examples::

            >>> rms_norm = nn.RMSNorm([2, 3])
            >>> input = torch.randn(2, 2, 3)
            >>> rms_norm(input)

        """

        __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
        normalized_shape: tuple[int, ...]
        eps: Optional[float]
        elementwise_affine: bool

        def __init__(
            self,
            normalized_shape: _shape_t,
            eps: Optional[float] = None,
            elementwise_affine: bool = True,
            device=None,
            dtype=None,
        ) -> None:
            factory_kwargs = {"device": device, "dtype": dtype}
            super().__init__()
            if isinstance(normalized_shape, numbers.Integral):
                # mypy error: incompatible types in assignment
                normalized_shape = (normalized_shape,)  # type: ignore[assignment]
            self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            if self.elementwise_affine:
                self.weight = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("weight", None)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            """
            Resets parameters based on their initialization used in __init__.
            """
            if self.elementwise_affine:
                init.ones_(self.weight)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Runs forward pass.
            """
            # return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
            return rms_norm(x, self.normalized_shape, self.weight, self.eps)

        def extra_repr(self) -> str:
            """
            Extra information about the module.
            """
            return (
                "{normalized_shape}, eps={eps}, "
                "elementwise_affine={elementwise_affine}".format(**self.__dict__)
            )

    nn.RMSNorm = RMSNorm    