"""
CLEVR-4: write latents.csv, CLIP RN50 image / caption features and augmented views, saved per split; with --tokens, their token features instead.
Run: python -m datasets.clevr4_features [--tokens]
"""
import argparse
import csv
import json
import os
import random

import pandas as pd
import torch
from omegaconf import OmegaConf
from torchvision import transforms as T

from datasets.base import RN50_IMAGE_TOKENS, TEXT_TOKEN_DIM, Images, clip_rn50, rn50_image_tokens, rn50_text_tokens
from datasets.clevr4 import TASKS, split_rows
from utils import REPO, set_seed


SHAPES_PLURAL = {'cone': 'cones',
                 'cube': 'cubes',
                 'cylinder': 'cylinders',
                 'diamond': 'diamonds',
                 'gear': 'gears',
                 'monkey': 'monkeys',
                 'sphere': 'spheres',
                 'star': 'stars',
                 'teapot': 'teapots',
                 'torus': 'tori'}

GENERIC_SHAPE = 'objects'

# feature variants: grayscale images, and captions with one factor dropped
IMAGE_VARIANTS = ['image', 'image_no_color']
TEXT_VARIANTS = {'text': [], 'text_no_color': ['color'], 'text_no_shape': ['shape'], 'text_no_texture': ['texture']}

# words a text augmentation may delete
FILLER = {'This', 'is', 'a', 'an', 'image', 'of', 'with', 'texture.'}


def caption(a, drop):
    """Template caption from the annotation, leaving out the factors in drop."""
    head = GENERIC_SHAPE if 'shape' in drop else SHAPES_PLURAL[a['shape']]
    if 'color' not in drop:
        head = f'{a["color"]} {head}'

    phrase = f'This is an image of {head}'
    if 'texture' not in drop:
        article = 'an' if a['texture'][0] in 'aeiou' else 'a'
        phrase += f' with {article} {a["texture"]} texture'

    return phrase + '.'


def augment_image(im, gray, preprocess, args):
    """k_aug views: crop / flip / jitter (after grayscale for image_no_color)."""
    aug = T.Compose(gray + [T.RandomResizedCrop(size=im.size[::-1], scale=(args.crop_min, 1.)),
                            T.RandomHorizontalFlip(),
                            T.RandomApply([T.ColorJitter(brightness=args.jitter, contrast=args.jitter)], p=0.8)])
    return torch.stack([preprocess(aug(im)) for _ in range(args.k_aug)])


def delete_filler_word(s):
    """Caption with one random filler word deleted."""
    words = s.split(' ')
    i = random.choice([j for j, w in enumerate(words) if w in FILLER])
    return ' '.join(words[:i] + words[i + 1:])


def write_latents(path, annots, ids):
    """Labels (color / shape / texture) and original split per image."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id'] + TASKS + ['split'])
        for k in ids:
            writer.writerow([k] + [annots[k][c] for c in TASKS] + [annots[k]['split']])
    print(f'wrote {path}')


@torch.no_grad()
def images(args, ids, rows):
    """CLIP features for each image variant."""
    model, preprocess, _ = clip_rn50(args.device)
    paths = [f'{args.data_root}/images/{k}.png' for k in ids]

    for image in IMAGE_VARIANTS:
        if all(os.path.exists(f'{args.out_dir}/{s}/{image}.pt') for s in rows):
            continue

        gray = [T.Grayscale(3)] if image == 'image_no_color' else []
        loader = torch.utils.data.DataLoader(Images(paths, T.Compose(gray + [preprocess])),
                                             batch_size=args.batch_size, num_workers=8)
        out = torch.cat([model.encode_image(x.to(args.device)).float().cpu() for x in loader])
        for s, r in rows.items():
            torch.save(out[r].clone(), f'{args.out_dir}/{s}/{image}.pt')
        print(f'wrote {args.out_dir}/<split>/{image}.pt')


@torch.no_grad()
def text(args, annots, ids, rows):
    """Captions (saved as .txt) and their CLIP features for each text variant."""
    model, _, tokenizer = clip_rn50(args.device)
    os.makedirs(f'{args.data_root}/text', exist_ok=True)

    for name, drop in TEXT_VARIANTS.items():
        if all(os.path.exists(f'{args.out_dir}/{s}/{name}.pt') for s in rows):
            continue

        lines = [caption(annots[k], drop) for k in ids]
        with open(f'{args.data_root}/text/{name}.txt', 'w') as f:
            f.write('\n'.join(lines))

        out = torch.cat([model.encode_text(tokenizer(lines[i:i + args.batch_size]).to(args.device)).float().cpu()
                         for i in range(0, len(ids), args.batch_size)])
        for s, r in rows.items():
            torch.save(out[r].clone(), f'{args.out_dir}/{s}/{name}.pt')
        print(f'wrote text/{name}.txt and {args.out_dir}/<split>/{name}.pt')


@torch.no_grad()
def augmented_views(args, annots, ids, rows):
    """
    k_aug views per sample: image crop / flip / jitter, caption with one filler word deleted.
    Computed over all images in one pass (fixes the random views), saved for train / val only.
    """
    model, preprocess, tokenizer = clip_rn50(args.device)
    paths = [f'{args.data_root}/images/{k}.png' for k in ids]

    for image in IMAGE_VARIANTS:
        name = f'{image}_aug{args.k_aug}_crop{args.crop_min}_jitter{args.jitter}'
        if all(os.path.exists(f'{args.out_dir}/{s}/{name}.pt') for s in ['train', 'val']):
            continue

        set_seed(args.data_seed)
        gray = [T.Grayscale(3)] if image == 'image_no_color' else []
        # batch size also decides which worker augments which image, so changing it changes the views
        loader = torch.utils.data.DataLoader(Images(paths, lambda im: augment_image(im, gray, preprocess, args)),
                                             batch_size=args.batch_size, num_workers=8)
        out = [model.encode_image(x.flatten(0, 1).to(args.device)).float().cpu().view(len(x), args.k_aug, -1)
               for x in loader]
        out = torch.cat(out).transpose(0, 1)
        for s in ['train', 'val']:
            torch.save(out[:, rows[s]].contiguous(), f'{args.out_dir}/{s}/{name}.pt')
        print(f'wrote {args.out_dir}/<split>/{name}.pt')

    for name, drop in TEXT_VARIANTS.items():
        if all(os.path.exists(f'{args.out_dir}/{s}/{name}_aug{args.k_aug}.pt') for s in ['train', 'val']):
            continue

        set_seed(args.data_seed)
        lines = [caption(annots[k], drop) for k in ids]

        texts = []
        for _ in range(args.k_aug):
            chunks = []
            for i in range(0, len(lines), args.batch_size):
                tokens = tokenizer([delete_filler_word(s) for s in lines[i:i + args.batch_size]]).to(args.device)
                chunks.append(model.encode_text(tokens).float().cpu())
            texts.append(torch.cat(chunks))

        texts = torch.stack(texts)
        for s in ['train', 'val']:
            torch.save(texts[:, rows[s]].contiguous(), f'{args.out_dir}/{s}/{name}_aug{args.k_aug}.pt')
        print(f'wrote {args.out_dir}/<split>/{name}_aug{args.k_aug}.pt')


@torch.no_grad()
def image_tokens(args, ids, rows):
    """
    RN50 grid tokens (fp16) for each image variant, and of the same k_aug views as augmented_views (train / val);
    computed over all images in one pass, as the pooled features.
    """
    model, preprocess, _ = clip_rn50(args.device)
    paths = [f'{args.data_root}/images/{k}.png' for k in ids]

    for image in IMAGE_VARIANTS:
        gray = [T.Grayscale(3)] if image == 'image_no_color' else []

        if not all(os.path.exists(f'{args.out_dir}/{s}/{image}_tokens.pt') for s in rows):
            out = torch.empty(len(paths), *RN50_IMAGE_TOKENS, dtype=torch.float16)
            i = 0
            loader = torch.utils.data.DataLoader(Images(paths, T.Compose(gray + [preprocess])),
                                                 batch_size=args.batch_size, num_workers=8)
            for x in loader:
                out[i:i + len(x)] = rn50_image_tokens(model, x.to(args.device)).cpu()
                i += len(x)
            for s, r in rows.items():
                torch.save(out[r].clone(), f'{args.out_dir}/{s}/{image}_tokens.pt')
            print(f'wrote {args.out_dir}/<split>/{image}_tokens.pt')

        name = f'{image}_tokens_aug{args.k_aug}_crop{args.crop_min}_jitter{args.jitter}'
        if all(os.path.exists(f'{args.out_dir}/{s}/{name}.pt') for s in ['train', 'val']):
            continue

        set_seed(args.data_seed)
        out = torch.empty(args.k_aug, len(paths), *RN50_IMAGE_TOKENS, dtype=torch.float16)
        i = 0
        loader = torch.utils.data.DataLoader(Images(paths, lambda im: augment_image(im, gray, preprocess, args)),
                                             batch_size=args.batch_size, num_workers=8)
        for x in loader:
            tokens = rn50_image_tokens(model, x.flatten(0, 1).to(args.device)).view(len(x), args.k_aug, *RN50_IMAGE_TOKENS)
            out[:, i:i + len(x)] = tokens.transpose(0, 1).cpu()
            i += len(x)
        for s in ['train', 'val']:
            torch.save(out[:, rows[s]].contiguous(), f'{args.out_dir}/{s}/{name}.pt')
        print(f'wrote {args.out_dir}/<split>/{name}.pt')


@torch.no_grad()
def text_tokens(args, annots, ids, rows):
    """RN50 text tokens (fp16, cut to the longest caption) for each text variant, and of the same k_aug views as augmented_views (train / val)."""
    model, _, tokenizer = clip_rn50(args.device)

    for name, drop in TEXT_VARIANTS.items():
        lines = [caption(annots[k], drop) for k in ids]

        if not all(os.path.exists(f'{args.out_dir}/{s}/{name}_tokens.pt') for s in rows):
            tokens = tokenizer(lines)
            length = int(tokens.argmax(-1).max()) + 1
            out = torch.empty(len(lines), length, TEXT_TOKEN_DIM, dtype=torch.float16)
            for i in range(0, len(lines), args.batch_size):
                out[i:i + args.batch_size] = rn50_text_tokens(model, tokens[i:i + args.batch_size].to(args.device))[:, :length].cpu()
            for s, r in rows.items():
                torch.save(out[r].clone(), f'{args.out_dir}/{s}/{name}_tokens.pt')
            print(f'wrote {args.out_dir}/<split>/{name}_tokens.pt')

        if all(os.path.exists(f'{args.out_dir}/{s}/{name}_tokens_aug{args.k_aug}.pt') for s in ['train', 'val']):
            continue

        set_seed(args.data_seed)
        tokens = torch.stack([tokenizer([delete_filler_word(s) for s in lines]) for _ in range(args.k_aug)])
        length = int(tokens.argmax(-1).max()) + 1
        out = torch.empty(args.k_aug, len(lines), length, TEXT_TOKEN_DIM, dtype=torch.float16)
        for k in range(args.k_aug):
            for i in range(0, len(lines), args.batch_size):
                out[k, i:i + args.batch_size] = rn50_text_tokens(model, tokens[k, i:i + args.batch_size].to(args.device))[:, :length].cpu()
        for s in ['train', 'val']:
            torch.save(out[:, rows[s]].contiguous(), f'{args.out_dir}/{s}/{name}_tokens_aug{args.k_aug}.pt')
        print(f'wrote {args.out_dir}/<split>/{name}_tokens_aug{args.k_aug}.pt')


def main():
    defaults = OmegaConf.load(f'{REPO}/configs/clevr4.yaml')
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=defaults['data_root'])
    p.add_argument('--features_dir', default=defaults['features_dir'])
    p.add_argument('--k_aug', type=int, default=defaults['k_aug'])
    p.add_argument('--crop_min', type=float, default=defaults['crop_min'])
    p.add_argument('--jitter', type=float, default=defaults['jitter'])
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--data_seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--tokens', action='store_true', help='token features (RePercENT) instead of pooled features')
    args = p.parse_args()
    args.out_dir = f'{args.features_dir}/{os.path.basename(args.data_root.rstrip("/"))}'   # one features folder per data_root
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{args.out_dir}/{split}', exist_ok=True)

    annots = json.load(open(f'{args.data_root}/clevr_4_annots.json'))
    ids = sorted(annots)
    write_latents(f'{args.data_root}/latents.csv', annots, ids)
    rows = split_rows(pd.read_csv(f'{args.data_root}/latents.csv'))
    if args.tokens:
        image_tokens(args, ids, rows)
        text_tokens(args, annots, ids, rows)
        return

    images(args, ids, rows)
    text(args, annots, ids, rows)
    augmented_views(args, annots, ids, rows)


if __name__ == '__main__':
    main()
