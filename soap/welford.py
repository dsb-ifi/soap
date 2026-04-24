import torch
import tqdm

from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader

class WelfordChanEstimator(nn.Module):

    '''Estimates first two moments of empirical distributions via Welford's method

    This implementation keeps track of the mean and covariance in a numerically stable way.
    '''

    def __init__(self, embed_dim:int, correction:float=1):
        super().__init__()
        self.embed_dim = embed_dim
        self._corr = correction
        self.register_buffer('n', torch.zeros(()))
        self.register_buffer('m1', torch.zeros(embed_dim))
        self.register_buffer('m2', torch.zeros(embed_dim, embed_dim))

    @property
    def correction(self) -> float:
        return self._corr
    
    @correction.setter
    def correction(self, value:float):
        self.set_correction(value)

    @property
    def mu(self) -> Tensor:
        return self.m1
    
    @property
    def cov(self) -> Tensor:
        cov = self.m2 / (self.n - self.correction)
        return cov
    
    @property
    def eigh(self):
        cov = self.cov
        eigval, eigvec = torch.linalg.eigh(cov)
        return eigval, eigvec
    
    def get_eigh_at_indices(self, indices:list[int]) -> tuple[Tensor, Tensor]:
        # Works for both negative and positive indices
        indices = [i % self.embed_dim for i in indices]
        eigval, eigvec = self.eigh
        return eigval[indices], eigvec[:, indices]
        
    def rank_to_indices(self, ranks:list[int]) -> list[int]:
        # Rank is inverse order of indices
        return [self.embed_dim - 1 - r for r in ranks]
    
    def get_eigh_at_ranks(self, ranks:list[int]) -> tuple[Tensor, Tensor]:
        indices = self.rank_to_indices(ranks)
        return self.get_eigh_at_indices(indices)

    def get_weights_and_biases(self):
        '''Returns a linear PCA projection and center biases

        Given an input x, the projection should be computed via x @ weights + biases
        For an nn.Linear, weights should be replaced by
        ```
        linear = nn.Linear(embed_dim, embed_dim, bias=True)
        weights, biases = self.get_weights_and_biases()
        linear.weight.data = weights.mT # Transpose, since nn.Linear weight is (out_features, in_features)
        linear.bias.data = biases
        ```
        '''
        _, weights = self.eigh
        biases = -self.m1 @ weights
        return weights, biases
    
    def get_truncated_weights_and_biases_at_indices(self, indices:list[int]):
        eigval, weights = self.get_eigh_at_indices(indices)
        biases = -self.m1 @ weights
        return weights, biases
    
    def get_truncated_weights_and_biases_at_ranks(self, ranks:list[int]):
        indices = self.rank_to_indices(ranks)
        return self.get_truncated_weights_and_biases_at_indices(indices)
    
    def get_linear(self, rank:bool=True, order:list[int]|None=None) -> nn.Linear:
        '''Retrieves PCA projection and biases packaged as an nn.Linear instance.

        Prefers rank ordering.

        Parameters
        ----------
        rank : bool
            Whether to use rank ordering (default: True).
        order : list[int]|None
            The order of indices to retrieve (default: None, which extracts all components).
        '''
        if order is None:
            order = list(range(self.embed_dim))
        getter = self.get_truncated_weights_and_biases_at_ranks if rank else self.get_truncated_weights_and_biases_at_indices
        weights, biases = getter(order)
        linear = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        linear.weight.data = weights.mT # Transpose, since nn.Linear weight is (out_features, in_features)
        linear.bias.data = biases
        return linear

    def set_correction(self, correction:float):
        assert correction >= 0, "Correction must be non-negative"
        self._corr = correction

    @torch.no_grad()
    def update(self, X:Tensor):
        E = X.shape[-1]
        assert E == self.embed_dim
        X = X.reshape(-1, E)
        m = torch.as_tensor(X.shape[0], dtype=X.dtype, device=X.device)
        s1_cur = X.sum(dim=0)
        m1_cur = s1_cur / m
        s2_cur = X.mT.mm(X)
        m2_cur = s2_cur - m * torch.outer(m1_cur, m1_cur)
        n_prev, m1_prev, m2_prev = self.n, self.m1, self.m2
        n_join = n_prev + m
        delta = m1_cur - m1_prev

        # Update
        self.m1 = m1_prev + delta * (m / n_join)
        self.m2 = m2_prev + m2_cur + torch.outer(delta, delta) * (n_prev * m / n_join)
        self.n = n_join

    def get_aggregated_patch_responses(
        self, 
        model:nn.Module,
        dataloader:DataLoader,
        device:torch.device,
        imgsize:int,
        patch_size:int, 
        embed_dim:int,
        path:str|None=None,
        binary:bool=True,
        imgindex:int=0,
        rank:bool=True,
        order:list[int]|None=None,
        num_globals:int=1,
        patch_indices:list[int]|None=None,
        forward_fn:str = 'forward_features'
    ) -> Tensor:
        assert imgsize % patch_size == 0
        np = imgsize // patch_size
        split = patch_indices # Use list of patches we want to extract
        if split is None:
            split = slice(num_globals, None) # Corresponds to outputs[num_global:]
        elif len(split) < np**2:
            raise ValueError(f"Not enough patches extracted: {len(split)} < {np**2}")

        lin = self.get_linear(rank=rank, order=order)
        responses = torch.zeros(np,np,embed_dim, device=device)
        cnt = 0
        model.eval()
        for batch in tqdm.tqdm(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[imgindex]
            inputs = batch.to(device)
            B = batch.shape[0]
            with torch.no_grad():
                outputs = getattr(model, forward_fn)(inputs)[:,split]
                cur_response = lin(outputs).view(B,np,np,-1)
                if binary:
                    cur_response = (cur_response > 0).float()
                responses += cur_response.sum(dim=0)
            cnt += B

        responses = responses / cnt

        if path is not None:
            torch.save(responses, path)

        return responses
    
    def get_aggregated_global_responses(
        self, 
        model:nn.Module,
        dataloader:DataLoader,
        device:torch.device,
        path:str|None=None,
        binary:bool=True,
        imgindex:int=0,
        rank:bool=True,
        order:list[int]|None=None,
        num_globals:int=1,
        global_indices:list[int]|None=None,
    ) -> Tensor:
        split = global_indices # Use list of patches we want to extract
        if split is None:
            split = slice(None, num_globals) # Corresponds to outputs[num_global:]

        lin = self.get_linear(rank=rank, order=order)
        responses = torch.zeros(num_globals,768, device=device)
        cnt = 0
        model.eval()
        for batch in tqdm.tqdm(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[imgindex]
            inputs = batch.to(device)
            B = batch.shape[0]
            with torch.no_grad():
                outputs = model.forward_features(inputs)[:,split]
                cur_response = lin(outputs).view(B,num_globals,-1)
                if binary:
                    cur_response = (cur_response > 0).float()
                responses += cur_response.sum(dim=0)
            cnt += B

        responses = responses / cnt

        if path is not None:
            torch.save(responses, path)

        return responses
    
    @classmethod
    def run_extraction(
        cls, 
        model:nn.Module, 
        dataloader:DataLoader, 
        device:torch.device, 
        embed_dim:int, 
        correction:float=1, 
        imgindex:int=0,
        extract_local:bool=True,
        num_globals:int=1,
        patch_indices:list[int]|None=None,
        forward_fn:str = 'forward_features'
    ) -> "WelfordChanEstimator":
        split = patch_indices # Use list of patches we want to extract
        if split is None:
            split = slice(None, num_globals) # Corresponds to outputs[:num_global]
            if extract_local:
                split = slice(num_globals, None) # Corresponds to outputs[num_global:]

        estimator = cls(embed_dim, correction).to(device)
        model.eval()
        for batch in tqdm.tqdm(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[imgindex]
            inputs = batch.to(device)
            with torch.no_grad():
                outputs = getattr(model, forward_fn)(inputs)[:,split]
            estimator.update(outputs)

        return estimator
    
    def serialize(self, path:str):
        torch.save({
            'embed_dim': self.embed_dim,
            'correction': self.correction,
            'n': self.n,
            'm1': self.m1,
            'm2': self.m2
        }, path)
    
    @classmethod
    def deserialize(cls, path:str) -> "WelfordChanEstimator":
        data = torch.load(path, map_location=torch.device('cpu'))
        estimator = cls(
            embed_dim=data['embed_dim'],
            correction=data['correction']
        )
        estimator.n = data['n']
        estimator.m1 = data['m1']
        estimator.m2 = data['m2']
        return estimator