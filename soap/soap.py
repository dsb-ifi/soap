import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .welford import WelfordChanEstimator
from typing import Literal, Callable
from functools import partial

def _get_cov(est: WelfordChanEstimator | Tensor) -> Tensor:
    if isinstance(est, WelfordChanEstimator):
        return est.cov
    return est


# def _max_scaler(x:Tensor, eps:float) -> Tensor:
#     mx = x.max()
#     return x / mx.clamp(min=eps)


# def _minmax_scaler(x:Tensor, eps:float=1e-12) -> Tensor:
#     mx, mn = x.max(), x.min()
#     return (x - mn) / (mx - mn).clamp(min=eps)


def entropy(prob:Tensor, dims:list[int]=[0,1], clamp_eps:float=1e-12) -> Tensor:
    p = prob.clamp(min=clamp_eps)
    return -(p*p.log() + (1-p)*(1-p).log())#.sum(dims)


def cross_entropy(p:Tensor, q:Tensor, dims:list[int]=[0,1], clamp_eps:float=1e-12) -> Tensor:
    p = p.clamp(min=clamp_eps)
    q = q.clamp(min=clamp_eps)
    return -(p*q.log() + (1-p)*(1-q).log()).sum(dims)


def jenson_shannon(p: Tensor, q: Tensor, dims:list[int]=[0,1], clamp_eps:float=1e-12) -> Tensor:
    m = .5 * (p + q)
    hm = entropy(m, dims=dims, clamp_eps=clamp_eps)
    hp = entropy(p, dims=dims, clamp_eps=clamp_eps)
    hq = entropy(q, dims=dims, clamp_eps=clamp_eps)
    return hm - (hp + hq) * .5


def kullback_leibler(p: Tensor, q: Tensor, dims:list[int]=[0,1], clamp_eps:float=1e-12) -> Tensor:
    hpq = cross_entropy(p, q, dims=dims, clamp_eps=clamp_eps)
    hp = entropy(p, dims=dims, clamp_eps=clamp_eps)
    return hpq - hp


def semantic_invariance(
    p: Tensor,
    q: Tensor,
    alpha: float = 2,
    gamma: float = 1,
    eps: float = 1e-12,
) -> Tensor:
    """ Semantic invariance score.
    """
    a = alpha
    g = gamma
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    num = (p*q + (1-p)*(1-q))**g
    denp = (p**a + (1-p)**a)**(1/a)
    denq = (q**a + (1-q)**a)**(1/a)
    out = 2*num / (denp + denq + eps)
    return out.clamp(eps,1-eps)


def semantic_invariance_score(
    p_data:Tensor, p_synth:Tensor,  
    dims:list[int]=[0,1], clamp_eps:float=1e-12, alpha:float=1.0
) -> Tensor:
    cs = semantic_invariance(p_data, p_synth, gamma=alpha).mean(dims).flip(-1)
    return cs

def semantic_invariance_score_transform(
    p_data:Tensor, p_synth:Tensor,
    dims:list[int]=[0,1], clamp_eps:float=1e-12, alpha:float=1.0,
    mu:float|None = 1.5, tau:float=0.2, threshold:float=0.75,
    capture:dict|None=None,
) -> Tensor:
    # Calculate scores
    scores = semantic_invariance_score(p_data, p_synth, dims, clamp_eps, alpha)

    if mu is None:
        mu = (scores > threshold).sum().item()
        if capture is not None:
            capture['mu'] = mu

    if mu <= 0:
        return torch.zeros_like(scores)

    # Sort and transform (ScoreTransform expects sorted scores)
    with torch.no_grad():
        t = ScoreTransform(mu=mu, tau=tau)
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        scores = t(sorted_scores)

        # Revert back to original order
        inv_idx = torch.argsort(sorted_idx)
        scores = scores[inv_idx]
    return scores


def semantic_invariance_projector_from_scores(
    scores:Tensor,
    cov_data:WelfordChanEstimator|Tensor, 
    clamp_eps:float=1e-12,
) -> Tensor:
    cov = _get_cov(cov_data)
    eigval, V = torch.linalg.eigh(cov)
    D = torch.diag_embed(scores)
    I = torch.eye(len(eigval), device=scores.device, dtype=scores.dtype)
    return I - V @ D @ V.mT


def semantic_invariance_projector(
    p_data:Tensor, p_synth:Tensor, cov_data:WelfordChanEstimator|Tensor,
    dims:list[int]=[0,1], clamp_eps:float=1e-12, alpha:float=1.0,
    semantic_invariance_score:Callable=semantic_invariance_score
) -> Tensor:
    scores = semantic_invariance_score(
        p_data=p_data, 
        p_synth=p_synth,
        dims=dims, 
        clamp_eps=clamp_eps, alpha=alpha)
    projector = semantic_invariance_projector_from_scores(scores, cov_data, clamp_eps)
    return projector


def truncated_invariance_projector(
    cov_data:WelfordChanEstimator|Tensor, trunc_indices:list[int],
    dims:list[int]=[0,1], clamp_eps:float=1e-12
) -> Tensor:
        cov = _get_cov(cov_data)
        eigval, eigvec = torch.linalg.eigh(cov)
        V = torch.stack([eigvec[...,idx] for idx in trunc_indices], -1)
        I = torch.eye(len(eigval), device=cov.device, dtype=cov.dtype)
        return I - V @ V.mT


class SOAP(nn.Module):
    """Semantically Orthogonal Artifact Projection (SOAP).

    Projects feature embeddings to suppress artifact directions identified
    by semantic invariance scoring across real and synthetic data distributions.
    """

    def __init__(self, projector:Tensor|None=None):
        super().__init__()
        if projector is None:
            projector = torch.eye(1, dtype=torch.float32)
        self.register_buffer("projector", projector)

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.projector.mT)
    
    def serialize(self,path:str) -> None:
        torch.save(self.projector, path)

    def update_projector(self, projector_new):
        # with torch.no_grad():
        #     self.projector.copy_(projector_new.to(self.projector.device))
        self.projector = projector_new.to(self.projector.device)
        # self.projector = projector_new.to(self.projector.device).detach()

    def update_from_scores(self, scores, cov_data:WelfordChanEstimator|Tensor, clamp_eps:float=1e-12):
        projector = semantic_invariance_projector_from_scores(scores, cov_data, clamp_eps)
        self.update_projector(projector)

    @classmethod
    def compute_from_scores(
        cls, scores:Tensor, cov_data:WelfordChanEstimator|Tensor, clamp_eps:float=1e-12
    ) -> "SOAP":
        projector = semantic_invariance_projector_from_scores(scores, cov_data, clamp_eps)
        return cls(projector)

    @classmethod
    def compute_from_data(
        cls, p_data:Tensor, p_synth:Tensor, cov_data:WelfordChanEstimator|Tensor,
        dims:list[int]=[0, 1], clamp_eps:float=1e-12, alpha:float=1.0, mu:float = 2.0, tau:float=0.05,
        score_version:Literal['scores', 'scaled']='scaled'
    ) -> "SOAP":
        if score_version == 'scores': 
            semantic_invariance_score = semantic_invariance_score
        elif score_version == 'scaled': 
            semantic_invariance_score = partial(semantic_invariance_score_transform, mu=mu, tau=tau)
        projector = semantic_invariance_projector(
            p_data, p_synth, cov_data, dims=dims, clamp_eps=clamp_eps, alpha=alpha,
            semantic_invariance_score=semantic_invariance_score
        )
        return cls(projector)
    
    @classmethod
    def from_precomputed(
        cls, path_response_data:str, path_response_synth:str, path_estimator_data:str, **kwargs
    ) -> "SOAP":
        e = WelfordChanEstimator.deserialize(path_estimator_data)
        p_data = torch.load(path_response_data, map_location=torch.device('cpu')).clamp(1e-12,1-1e-12)
        p_synth = torch.load(path_response_synth, map_location=torch.device('cpu')).clamp(1e-12,1-1e-12)
        return cls.compute_from_data(p_data, p_synth, e, **kwargs)
    
    @classmethod
    def from_modelname(cls, modelname:str, path:str, binary:bool=True, **kwargs
    ) -> "SOAP":
        if binary:
            path_response_data = os.path.join(path, f"{modelname}_agg_patch_responses.pth")
            path_response_synth = os.path.join(path, f"{modelname}_agg_patch_responses_synth.pth")
        else:
            path_response_data = os.path.join(path, f"{modelname}_agg_patch_softresponses.pth")
            path_response_synth = os.path.join(path, f"{modelname}_agg_patch_softresponses_synth.pth")
        path_estimator_data = os.path.join(path, f"{modelname}_cov.pth")
        return cls.from_precomputed(
            path_response_data, path_response_synth, path_estimator_data, **kwargs)

    @classmethod
    def deserialize(cls, path:str) -> "SOAP":
        projector = torch.load(path, map_location=torch.device('cpu'))
        return cls(projector)
    
    @classmethod
    def manual_truncation(
        cls, cov_data:WelfordChanEstimator|Tensor, trunc_indices:list[int],
    ) -> "SOAP":
        """ Manually select which principal components to supress. Used in testing.
        """
        projector = truncated_invariance_projector(cov_data, trunc_indices)
        return cls(projector)


def softplus_inv(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.exp().sub(1).log()


def sigmoid_inv(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return torch.logit(x)
    

def locscale_sigmoid(x, m, t):
    # m = x.new_tensor(m)
    # t = x.new_tensor(t)
    n = ((m-x)/t).sigmoid()
    d = (m/t).sigmoid()
    return n / d

class ScoreTransform(nn.Module):

    def __init__(self, mu:float=2.5, tau:float=0.2):
        super().__init__()
        self._mu = nn.Parameter(softplus_inv(mu))
        self._tau = nn.Parameter(softplus_inv(tau))

    @property
    def mu(self):
        return F.softplus(self._mu)

    @property
    def tau(self):
        return F.softplus(self._tau)

    def forward(self, sorted_scores):
        s = sorted_scores
        r = torch.arange(len(s), device=s.device, dtype=s.dtype)
        flter = torch.sigmoid((self.mu - 0.5 - r) / self.tau)
        return flter * s