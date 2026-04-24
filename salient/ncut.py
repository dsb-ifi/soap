import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy import ndimage
from typing import Literal

def ncut(feats, dims, scales, init_image_size, guide=None,
         tau = 0, eps=1e-5, foregound_selection='pca',
         no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    eigenvec = second_smallest_eigenvector(feats, tau, eps, no_binary_graph)

    # Bi-partition with avg value eigenvector
    avg = np.sum(eigenvec) / len(eigenvec)
    bipartition = eigenvec > avg

    # Determine foregound partition
    bipartition, seed = select_foreground(eigenvec, bipartition, guide, method=foregound_selection)
    bipartition = bipartition.reshape(dims).astype(float)

    # Select object component
    objects, bipartition = detect_object(bipartition, seed, dims)
    
    # Interpolate the bipartition and eigenvec to image size
    bipartition = torch.from_numpy(bipartition).to('cuda')
    bipartition = F.interpolate(bipartition.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
    eigvec = eigenvec.reshape(dims) 
    eigvec = torch.from_numpy(eigvec).to('cuda')
    eigvec = F.interpolate(eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()

    return  seed, bipartition.cpu().numpy(), eigvec.cpu().numpy()

def second_smallest_eigenvector(feats, tau, eps=1e-5, no_binary_graph=False):
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0,1) @ feats)
    A = A.cpu().numpy()
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)

    # Get second smallest eigenvector
    _, eigenvec = eigh(D-A, D, subset_by_index=[1,1])
    return eigenvec

def select_foreground(eigenvec, bipartition, guide=None, method:Literal['pca', 'vmax']='vmax'):
    if method=='vmax':
        seed = np.argmax(np.abs(eigenvec)) # Patch with max absval of eigvec
        if bipartition[seed] != 1:
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
    elif method=='pca':
        seed = np.argmax(guide)
        if bipartition[seed] != 1:
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
    return bipartition, seed

def detect_object(bipartition, seed, dims):
    """
    Extract the component corresponding to the seed patch. Among connected components 
    in the bipartition mask, select the one corresponding to the seed patch.
    """
    objects, num_objects = ndimage.label(bipartition) # Separate into components
    cc = objects[np.unravel_index(seed, dims)]        # Select component containing the seed patch
    mask = np.where(objects == cc)                    # Object mask
    bipartition = np.zeros(dims)
    bipartition[mask[0],mask[1]] = 1
    return objects, bipartition