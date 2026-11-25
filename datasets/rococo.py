import json

import torch

from .base import Dataset, index_batches, load_features

ROCOCO_SETS = ['rand_voca', 'same_concept', 'diff_concept', 'danger']
DISTRACTORS = ['mix_0.9', 'mix_0.8', 'patch_0.9', 'patch_0.8']
MIX_SEEDS = [1, 10, 100]

# feature files ('parts') used for evaluation
TXT_PARTS = ['test'] + [f'rococo_{n}' for n in ROCOCO_SETS]
IMG_PARTS = ['test'] + [f'{d}_seed{s}' for d in DISTRACTORS for s in MIX_SEEDS]


class RoCOCO(Dataset):
    """
    COCO (Karpathy splits) CLIP ViT-B/32 features (args.tokens: token features; args.pre_projection: before CLIP's projections), several captions per image: each epoch samples one caption per image.
    Views: augmented image crops (x_aug[split] is image only), and a different caption of the same image as the text view.
    Images (captions): train 113,287 (566,747) / val 5,000 (25,010) / test 5,000 (25,010).
    Eval only: each rococo_* set = the 5,000 test images with 50,010 captions (25,000 adversarial);
    each distractor part = 5,000 extra images.
    """
    def __init__(self, args, views=False):
        super().__init__(args, views)
        self.loaded = {}
        self.x = {s: (self.features(s, 'image'), self.features(s, 'text')) for s in ['train', 'val']}
        self.dims = (self.x['train'][0].shape[-1], self.x['train'][1].shape[-1])

        # txt2img / img2txt / wrongtext indices of every part with captions
        self.meta = {p: json.load(open(part_path(args.features_dir, p, 'meta.json'))) for p in ['train', 'val'] + TXT_PARTS}
        self.sample_train = caption_sampler(self.meta['train']['txt2img'], len(self.x['train'][0]), self.x['train'][0].device)
        self.val_caption = None

        if views:
            suffix = '_tokens' if args.tokens else ''
            self.x_aug = {split: load_features(f'{args.features_dir}/{split}/image{suffix}_aug{args.k_aug}.pt', args) for split in ['train', 'val']}

    def features(self, part, modality):
        """Features of one file (split, rococo_* or distractor part), loaded once."""
        if (part, modality) not in self.loaded:
            suffix = '_tokens' if self.args.tokens else '_pre_projection' if getattr(self.args, 'pre_projection', False) else ''
            name = f'{modality}{suffix}.pt'
            self.loaded[part, modality] = load_features(part_path(self.args.features_dir, part, name), self.args)
        return self.loaded[part, modality]

    def batches(self, split, shuffle, views=False):
        # val captions sampled once, so val loss is comparable across epochs
        if self.val_caption is None:
            n_val = len(self.x['val'][0])
            self.val_caption = caption_sampler(self.meta['val']['txt2img'], n_val, self.x['val'][0].device)(views)

        image, text = self.x[split]
        caption = self.sample_train(views) if split == 'train' else self.val_caption

        idx = torch.arange(len(image), device=image.device)
        for b in index_batches(idx, self.args.batch_size, shuffle, drop_last=split == 'train'):
            if views:
                k = torch.randint(self.args.k_aug, (len(b),), device=b.device)
                batch = image[b], text[caption[0][b]], self.x_aug[split][k, b], text[caption[1][b]]
            else:
                batch = image[b], text[caption[b]]
            yield tuple(t.to(self.args.dev) for t in batch)


def part_path(root, part, name):
    """train / val / test files sit in their split folder; rococo_* and distractor parts are test data, in test/."""
    return f'{root}/{part}/{name}' if part in ('train', 'val', 'test') else f'{root}/test/{part}_{name}'


def caption_sampler(txt2img, n_img, dev):
    """sample() -> one random caption index per image; with views, two different captions per image."""
    txt2img = torch.as_tensor(txt2img, device=dev)
    order = torch.argsort(txt2img, stable=True)
    counts = torch.bincount(txt2img, minlength=n_img)
    offsets = torch.cumsum(counts, 0) - counts

    def sample(views=False):
        r = (torch.rand(n_img, device=dev) * counts).long()
        if not views:
            return order[offsets + r]

        other = (r + 1 + (torch.rand(n_img, device=dev) * (counts - 1)).long()) % counts
        return order[offsets + r], order[offsets + other]

    return sample
