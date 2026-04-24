import torch
import torchvision as tv 
import os
import argparse
import torch.nn as nn
from torch.utils.data import ConcatDataset

from get_models import get_dense_backbone
from soap import SOAP, WelfordChanEstimator
from soap.synth import SynthesizedDataSet
from types import MethodType


def get_data(dataset, datafolder, imgproc):
    data = ... # Load Dataset e.g. ImageNet. Remember to apply normalization.
    return data

def fit_WCE(model, dataloader, modelname, device, forward_fn='forward', num_global_tokens:int=0, patch_indices=None, dump_dir='weights'):
    embed_dim = model.embed_dim
    os.makedirs(dump_dir, exist_ok=True)
    filename = os.path.join(dump_dir, f'{modelname}_cov.pth')
    if os.path.exists(filename):
        # Read from file
        cov_data = WelfordChanEstimator.deserialize(filename).to(device)
    else:
        # Estimate from data
        print("Fitting WelfordChanEstimator")
        cov_data = WelfordChanEstimator.run_extraction(
            model, dataloader, device, embed_dim, 
            num_globals=num_global_tokens,
            patch_indices=patch_indices,
            forward_fn=forward_fn
        )
        cov_data.serialize(filename)
    
    return cov_data

def fit_projector(model, cov_data:WelfordChanEstimator, dataloader, dataloader_synth, modelname, imgsize,
                patch_size, device, forward_fn='forward', num_global_tokens:int=0, patch_indices=None,
                dump_dir='weights', soft_responses:bool=False):
    if soft_responses:
        responses_synth = os.path.join(dump_dir, f"{modelname}_agg_patch_softresponses_synth.pth")
        responses_real = os.path.join(dump_dir, f"{modelname}_agg_patch_softresponses.pth")
    else:
        responses_synth = os.path.join(dump_dir, f"{modelname}_agg_patch_responses_synth.pth")
        responses_real = os.path.join(dump_dir, f"{modelname}_agg_patch_responses.pth")
    
    embed_dim = model.embed_dim
    os.makedirs(dump_dir, exist_ok=True)
    if not os.path.exists(responses_synth):
        # Aggregated responses for synth data
        print("Calculating aggregates response for synth data")
        cov_data.get_aggregated_patch_responses(
            model, dataloader_synth, device, imgsize, patch_size, embed_dim,
            responses_synth,
            num_globals=num_global_tokens,
            patch_indices=patch_indices,
            forward_fn=forward_fn,
            binary=(not soft_responses)
        )

    if not os.path.exists(responses_real):
        # Aggregated responses for real data
        print("Calculating aggregates response for real data")
        cov_data.get_aggregated_patch_responses(
            model, dataloader, device, imgsize, patch_size, embed_dim,
            responses_real,
            num_globals=num_global_tokens,
            patch_indices=patch_indices,
            forward_fn=forward_fn,
            binary=(not soft_responses)
        )

    # Calculate projection
    wce_file = os.path.join(dump_dir, f'{modelname}_cov.pth')
    projector = SOAP.from_precomputed(
        responses_real,
        responses_synth,
        wce_file
    )

    if soft_responses:
        projector_file = os.path.join(dump_dir, f"{modelname}_projector_from_softresponses.pth")
    else:
        projector_file = os.path.join(dump_dir, f"{modelname}_projector.pth")
    projector.serialize(projector_file)

    return projector

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', type=str)
    parser.add_argument('--dataset', type=str, default='IN1k')
    parser.add_argument('--backbone', type=str, default='dinov2_base')
    parser.add_argument('--dump_dir', type=str, default='weights')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--features', type=str, default='out')
    parser.add_argument('--vit_layer', type=int, default=-1)
    parser.add_argument('--soft_responses', action='store_true', default=False)
    args = parser.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    # Get real and synthetic data
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    imgsize = 224
    proc = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop((imgsize, imgsize), (1., 1.)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean, std),
    ])
    data = get_data(args.dataset, args.datafolder, proc)
    dataloader = torch.utils.data.DataLoader(data, args.batch_size, num_workers=4, prefetch_factor=2, pin_memory=True)
    dataset_length = len(data)
    dataloader_synth = SynthesizedDataSet(size=imgsize, channels=3, batch_size=args.batch_size, length=dataset_length)
    device = torch.device('cuda')

    # Get model
    model = get_dense_backbone(args.backbone)
    num_global_tokens = 0 # These are removed already
    patch_indices = None
    patch_size = model.patch_size
    def forward_fn(self, img, vit_feat:str=args.features, vit_layer:int=args.vit_layer):
        return self.forward(img, vit_feat, vit_layer)
    model.forward_fn = MethodType(forward_fn, model)
    forward_fn = 'forward_fn' # Overwrite
    model.to(device).eval()
    embed_dim = model.embed_dim

    # Fit WCE
    cov_data = fit_WCE(
        model, dataloader, args.backbone, device,
        forward_fn=forward_fn,
        num_global_tokens=num_global_tokens,
        patch_indices=patch_indices,
        dump_dir=args.dump_dir)

    # Fit projector
    projector = fit_projector(
        model, cov_data, dataloader, dataloader_synth,
        args.backbone, imgsize, patch_size, device,
        forward_fn=forward_fn,
        num_global_tokens=num_global_tokens,
        patch_indices=patch_indices,
        dump_dir=args.dump_dir,
        soft_responses=args.soft_responses)