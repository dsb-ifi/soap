"""
Evaluate zero-shot salient segmentation
"""

import numpy as np
import argparse
import os
import json
import numpy as np
import torchvision as tv
from functools import partial

from soap import SOAP, WelfordChanEstimator
from get_models import *
from salient.bilateral import *
from salient.ncut import *
from salient.metrics import *
from salient.util import *

def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)

def none_or_float(v):
    if v.lower() == "none":
        return None
    return float(v)

def get_data(dataset:str, datafolder:str):
    if dataset == 'ECSSD':
        return ... # Get the ECSSD dataset from https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html
    else:
        raise NotImplementedError(f'Dataset {dataset} is unknown')

def get_guide_response(feat, W, b, flip=True):
    m = feat @ W.mT + b
    if flip: m = -m
    return m

if __name__=='__main__':
    parser = argparse.ArgumentParser("Evaluate TokenCut")
    parser.add_argument('--dataset', type=str, default='ECSSD', choices=['ECSSD'])
    parser.add_argument('--datafolder', type=str)
    parser.add_argument('--in1k_normalize', action='store_true', default=True)
    parser.add_argument('--no-in1k_normalize', action='store_false', dest='in1k_normalize')
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--tokencut_tau', type=float, default=0.3)
    parser.add_argument('--sigma_spatial', type=int, default=16)
    parser.add_argument('--sigma_luma', type=int, default=16)
    parser.add_argument('--sigma_chroma', type=int, default=8)
    parser.add_argument('--features', type=str, default='out', choices=['k', 'q', 'v', 'kqv', 'out'])
    parser.add_argument('--n_visualize', type=int, default=25,
                        help = 'Number of images to visualize (save)')
    parser.add_argument('--dump_dir', type=str, default='salient_results')
    parser.add_argument('--checkpoint_folder', type=str, default='weights')
    parser.add_argument('--pca_guide_dim', type=none_or_int, default=None)
    parser.add_argument('--flip_guide', action='store_true', default=False)
    parser.add_argument('--suppress_pca_components', type=int, nargs='+', default=None)
    parser.add_argument('--semantic_invariance_projector', action='store_true', default=False)
    parser.add_argument('--score_version', type=str, choices=['scores', 'scaled'],  default='scaled')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--mu', type=none_or_float, default=2.0)
    parser.add_argument('--tau', type=none_or_float, default=0.05)
    args = parser.parse_args()

    # Check args
    if args.semantic_invariance_projector:
        msg = f'Cannot do manual truncation with --suppress_pca_components and use --semantic_invariance_projector at the same time.'
        assert args.suppress_pca_components == None, msg
    if args.score_version != 'scaled': 
        args.mu = None
        args.tau = None
    
    # Print args
    for k, v in vars(args).items(): print(f"{k}: {v}")

    os.makedirs(os.path.join(args.dump_dir, 'images'), exist_ok=True)
    data = get_data(args.dataset, args.datafolder)
    backbone = get_dense_backbone(args.backbone).cuda().eval()
    patch_size = backbone.patch_size

    # Semantic invariance projector from SI scores
    if args.semantic_invariance_projector:
        project = SOAP.from_modelname(
            args.backbone, args.checkpoint_folder, alpha=args.gamma, mu=args.mu, tau=args.tau,
            score_version=args.score_version
            ).cuda()

    # Projector from manual selection of principal components to suppress
    wce = WelfordChanEstimator.deserialize(os.path.join(args.checkpoint_folder, f"{args.backbone}_cov.pth")).cuda()
    if args.suppress_pca_components is not None:
        project = SOAP.manual_truncation(wce, args.suppress_pca_components)

    # Salient guide (optional)
    if args.pca_guide_dim is not None:
        W, b = wce.get_truncated_weights_and_biases_at_indices([args.pca_guide_dim])
        get_guide = partial(get_guide_response, W=W.mT, b=b, flip=args.flip_guide)
    
    mask_raw = []
    mask_BS = []
    mask_BS_binary = []
    seg_labels = []

    for i, (img, seg) in enumerate(data):
        seg_labels.append(seg)
        
        I_resize, w, h, feat_w, feat_h = resize_pil(img, patch_size)
        if args.in1k_normalize:
            trans = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225)),])
        else: trans = tv.transforms.ToTensor()
        tensor = trans(I_resize).unsqueeze(0)

        # Get patch features from model
        feat = backbone(tensor.cuda(), vit_feat=args.features)[0]

        # Get guide from pca split
        if args.pca_guide_dim is not None:
            guide = get_guide(feat).cpu().numpy()
        else:
            guide = None

        # Suppress pca components (SI score)
        if args.semantic_invariance_projector:
            feat = project(feat)

        # Suppress pca components (manual)
        elif args.suppress_pca_components is not None:
            feat = project(feat.unsqueeze(0))
            feat = feat.squeeze()

        # NCUT
        fg_select = 'vmax' if args.pca_guide_dim == None else 'pca' 
        seed, bipartition, eigvec = ncut(
            feat.mT, dims=[feat_h, feat_w], 
            scales=[patch_size, patch_size], init_image_size=[h,w], 
            guide=guide, foregound_selection=fg_select, tau=args.tokencut_tau)
        mask_raw.append(bipartition)

        # Bilateral solver for edge refinement
        output_solver, binary_solver = bilateral_solver_output(
            img, bipartition, 
            sigma_spatial = args.sigma_spatial, 
            sigma_luma = args.sigma_luma, 
            sigma_chroma = args.sigma_chroma
        )
        mask_BS.append(output_solver)

        # Check if mask needs to be flipped
        mask1 = torch.from_numpy(bipartition).cuda()
        mask2 = torch.from_numpy(binary_solver).cuda()
        if IoU(mask1, mask2) < 0.5:
            binary_solver = binary_solver * -1
        mask_BS_binary.append(binary_solver)

        if i < args.n_visualize:
            org = np.array(img)
            mask_color_compose(org, bipartition).save(os.path.join(args.dump_dir, 'images', f'{i}_tokencut.jpg'))
            mask_color_compose(org, output_solver).save(os.path.join(args.dump_dir, 'images', f'{i}_tokencut_BS.jpg'))
            mask_color_compose(org, binary_solver).save(os.path.join(args.dump_dir, 'images', f'{i}_tokencut_BS_binary.jpg'))
            mask_color_compose(org, seg).save(os.path.join(args.dump_dir, 'images', f'{i}_gt_seg.jpg'))
            img.save(os.path.join(args.dump_dir, 'images', f'{i}_original.jpg'))

        print(f'{i+1} / {len(data)}')

    # TokenCut eval
    metrics_tokencut = metrics(mask_raw, seg_labels)
    print('\n', metrics_tokencut)

    # TokenCut + bilateral solver eval
    metrics_tokencut_BS = metrics(mask_BS, seg_labels)
    print('\n', metrics_tokencut_BS)

    # TokenCut + bilateral solver (binary mask) eval
    metrics_tokencut_BS_binary = metrics(mask_BS_binary, seg_labels)
    print('\n', metrics_tokencut_BS_binary)

    # Save results
    with open(os.path.join(args.dump_dir, 'salient_segmentation_results.json'), 'w') as f:
        json.dump({
            "args" : vars(args),
            "Tokencut" : metrics_tokencut,
            "Tokencut_BS" : metrics_tokencut_BS,
            "Tokencut_BS_binary" : metrics_tokencut_BS_binary
        }, f, indent=4)
    print(f"Results saved to {os.path.join(args.dump_dir, 'salient_segmentation_results.json')}")


        