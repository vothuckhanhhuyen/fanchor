"""
Synthetic task: generate latents, one view per modality and one label per latent; n_modalities=2 is the original DisentangledSSL data.
Run: python -m datasets.synthetic_data [--n_modalities 3]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from datasets.synthetic import LATENTS
from utils import load_defaults


N_SAMPLES = 90000
LATENT_DIM = 50
INPUT_DIM = 100


def main():
    defaults = load_defaults('synthetic', 'fanchor')
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=defaults['data_root'])
    p.add_argument('--data_seed', type=int, default=0)
    p.add_argument('--n_modalities', type=int, default=defaults['n_modalities'])
    args = p.parse_args()
    os.makedirs(args.data_root, exist_ok=True)

    rng = np.random.RandomState(args.data_seed)
    modalities = range(1, args.n_modalities + 1)
    z = {k: rng.multivariate_normal(np.zeros(LATENT_DIM), np.eye(LATENT_DIM) * 0.5, size=N_SAMPLES) for k in ['zs'] + [f'z{m}' for m in modalities]}

    x = {f'x{m}': np.concatenate([z[f'z{m}'], z['zs']], 1) @ rng.uniform(-1.0, 1.0, (2 * LATENT_DIM, INPUT_DIM)) for m in modalities}

    labels = {}
    for mlp_seed, k in enumerate(LATENTS + [f'z{m}' for m in modalities[2:]]):
        torch.manual_seed(mlp_seed)
        label_mlp = nn.Sequential(nn.Linear(LATENT_DIM, 100), nn.ReLU(), nn.Linear(100, 1))
        for layer in label_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        with torch.no_grad():
            logit = label_mlp(torch.Tensor(z[k])).numpy()
        prob = 1 / (1 + np.exp(-(logit + rng.normal(0, 0.1, logit.shape))))
        labels[f'y_{k}'] = (prob >= np.median(prob)).astype(np.int64).reshape(-1)

    path = f'{args.data_root}/synthetic_{args.n_modalities}mod.npz'
    np.savez(path, **x, **z, **labels)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
