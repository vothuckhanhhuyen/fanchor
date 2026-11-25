import json

import numpy as np
import pandas as pd

from utils import THIRD_PARTY
from .base import Dataset, load_features

VGGSOUNDER_DATA = f'{THIRD_PARTY}/VGGSounder/vggsounder/data'
CLASSES = pd.read_csv(f'{VGGSOUNDER_DATA}/classes.csv')['display_name'].tolist()   # 309 classes, VGGSounder index order


class VGGSound(Dataset):
    """
    PE-AV (base, 16 frames) video / audio features (args.tokens: token features): video takes the image slot, audio the text slot.
    One VGGSound class per clip (index into CLASSES); ids match the VGGSounder video_id.
    Views (train / val): k_aug augmented PE-AV embeddings per clip and modality, x_aug[split] = (video, audio), each [k_aug, n, dim].
    train 164,389 / val 18,225 / test 15,446 (all 12,014 VGGSounder clips are in test); train / val split by YouTube video.
    """
    def __init__(self, args, views=False):
        super().__init__(args, views)
        suffix = '_tokens' if args.tokens else ''
        self.x = {s: (load_features(f'{args.features_dir}/{s}/video{suffix}.pt', args),
                      load_features(f'{args.features_dir}/{s}/audio{suffix}.pt', args)) for s in ['train', 'val', 'test']}
        self.dims = (self.x['train'][0].shape[-1], self.x['train'][1].shape[-1])

        meta = {s: json.load(open(f'{args.features_dir}/{s}/meta.json')) for s in ['train', 'val', 'test']}
        class_index = {c: i for i, c in enumerate(CLASSES)}
        self.ids = {s: m['ids'] for s, m in meta.items()}
        self.labels = {s: np.array([class_index[c] for c in m['labels']]) for s, m in meta.items()}

        if views:
            self.x_aug = {s: (load_features(f'{args.features_dir}/{s}/video{suffix}_aug{args.k_aug}.pt', args),
                              load_features(f'{args.features_dir}/{s}/audio{suffix}_aug{args.k_aug}.pt', args)) for s in ['train', 'val']}
