import os

import numpy as np
import pandas as pd

from .base import Dataset, load_features

TASKS = ['color', 'shape', 'texture']   # probed factors
VAL_FRAC = 0.1   # share of the original train split held out as val


class CLEVR4(Dataset):
    """
    CLIP RN50 image / caption features (args.tokens: token features); args.image / args.text pick the variant (e.g. image_no_color, text_no_shape).
    clevr_4_100k: train 72,011 / val 8,001 / test 19,988.
    """
    def __init__(self, args, views=False):
        super().__init__(args, views)
        labels = pd.read_csv(f'{args.data_root}/latents.csv')
        self.labels = {s: labels.iloc[r] for s, r in split_rows(labels).items()}

        d = f'{args.features_dir}/{os.path.basename(args.data_root.rstrip("/"))}'   # one features folder per data_root
        suffix = '_tokens' if args.tokens else ''
        self.x = {s: (load_features(f'{d}/{s}/{args.image}{suffix}.pt', args), load_features(f'{d}/{s}/{args.text}{suffix}.pt', args))
                  for s in ['train', 'val', 'test']}
        self.dims = (self.x['train'][0].shape[-1], self.x['train'][1].shape[-1])

        if views:
            self.x_aug = {s: (load_features(f'{d}/{s}/{args.image}{suffix}_aug{args.k_aug}_crop{args.crop_min}_jitter{args.jitter}.pt', args),
                              load_features(f'{d}/{s}/{args.text}{suffix}_aug{args.k_aug}.pt', args)) for s in ['train', 'val']}


def split_rows(labels):
    """Row indices per split: val = VAL_FRAC of the original train split (seed 0), test = the original val split."""
    pool = np.flatnonzero(labels.split == 'train')
    val = np.zeros(len(labels), bool)
    val[np.random.default_rng(0).choice(pool, round(VAL_FRAC * len(pool)), replace=False)] = True
    masks = {'train': (labels.split == 'train').values & ~val, 'val': val, 'test': (labels.split == 'val').values}
    return {s: np.flatnonzero(mask) for s, mask in masks.items()}
