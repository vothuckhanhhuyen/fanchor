import numpy as np
import torch

from .base import Dataset

LATENTS = ['z1', 'zs', 'z2']   # private to x1, shared, private to x2 (+ z3, ... for more modalities); one binary label per latent
TEST_FRAC = 0.2   # as the original synthetic task
VAL_FRAC = 0.1   # share of the train split held out as val


class Synthetic(Dataset):
    """
    DisentangledSSL synthetic task, oracle inputs: [z_m, zs] per modality (args.n_modalities), latents 50-d.
    90,000 samples: train 64,800 / val 7,200 / test 18,000.
    """
    def __init__(self, args, views=False):
        super().__init__(args, views)
        d = np.load(f'{args.data_root}/synthetic_{args.n_modalities}mod.npz')
        modalities = range(1, args.n_modalities + 1)
        inputs = tuple(np.concatenate([d[f'z{m}'], d['zs']], 1) for m in modalities)
        latents = LATENTS + [f'z{m}' for m in modalities[2:]]

        rows = split_rows(len(d['x1']))
        self.x = {s: tuple(torch.from_numpy(v[r]).float().to(args.dev) for v in inputs) for s, r in rows.items()}
        self.latents = {s: {z: d[z][r] for z in latents} for s, r in rows.items()}
        self.labels = {s: {z: d[f'y_{z}'][r] for z in latents} for s, r in rows.items()}
        self.dims = tuple(v.shape[1] for v in self.x['train'])


def split_rows(n):
    """Row indices per split: test = the original 20% (torch random_split, seed 0), val = VAL_FRAC of the rest (seed 0)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).numpy()
    n_train = int((1 - TEST_FRAC) * n)
    pool, test = perm[:n_train], perm[n_train:]
    val = np.zeros(n, bool)
    val[np.random.default_rng(0).choice(pool, round(VAL_FRAC * len(pool)), replace=False)] = True
    return {'train': pool[~val[pool]], 'val': pool[val[pool]], 'test': test}
