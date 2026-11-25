"""
RoCOCO: metadata, CLIP ViT-B/32 image / caption features of every part, and augmented image views; with --tokens, their token features instead;
with --pre_projection, the features before CLIP's projections (and the projections themselves, for finetuned_proj_clip).
Run: python -m datasets.rococo_features [--tokens | --pre_projection]
"""
import argparse
import json
import os

import clip
import torch
from omegaconf import OmegaConf
from torchvision import transforms as T

from datasets.base import (TEXT_TOKEN_DIM, VITB32_IMAGE_TOKENS, Images, clip_vitb32, vitb32_image_pre_projection, vitb32_image_tokens,
                           vitb32_text_pre_projection, vitb32_text_tokens)
from datasets.rococo import DISTRACTORS, MIX_SEEDS, ROCOCO_SETS, part_path
from utils import REPO, set_seed

# train / val / test, the RoCOCO caption sets (test images, adversarial captions) and the distractor image sets
PARTS = ['train', 'val', 'test'] + [f'rococo_{n}' for n in ROCOCO_SETS] + [f'{d}_seed{s}' for d in DISTRACTORS for s in MIX_SEEDS]


def augment_image(im, preprocess, k_aug):
    """k_aug views: crop / flip / jitter, as clevr4."""
    aug = T.Compose([T.RandomResizedCrop(size=im.size[::-1], scale=(0.8, 1.)),
                     T.RandomHorizontalFlip(),
                     T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.8)])
    return torch.stack([preprocess(aug(im)) for _ in range(k_aug)])


def karpathy_part(split, root):
    """COCO Karpathy split: image list, captions, txt2img (and img2txt for val / test)."""
    annotations = json.load(open(f'{root}/annotations/coco_karpathy_{split}.json'))
    if split == 'train':
        img_idx = {}
        for a in annotations:
            img_idx.setdefault(a['image'], len(img_idx))
        return {'images': list(img_idx), 'texts': [a['caption'].strip() for a in annotations],
                'txt2img': [img_idx[a['image']] for a in annotations]}

    texts, txt2img, img2txt = [], [], []
    for i, a in enumerate(annotations):
        img2txt.append(list(range(len(texts), len(texts) + len(a['caption']))))
        texts += [c.strip() for c in a['caption']]
        txt2img += [i] * len(a['caption'])
    return {'images': [a['image'] for a in annotations], 'texts': texts, 'txt2img': txt2img, 'img2txt': img2txt}


def rococo_part(name, root):
    """RoCOCO caption set: the first captions of each image are true, the rest adversarial (wrongtext)."""
    annotations = json.load(open(f'{root}/annotations/{name}.json'))
    texts, img2txt, wrongtext = [], [], []

    for a in annotations:
        n_true = {10: 5, 11: 6}.get(len(a['caption']), len(a['caption']))
        ids = list(range(len(texts), len(texts) + len(a['caption'])))
        img2txt.append(ids[:n_true])
        wrongtext += ids[n_true:]
        texts += [c.strip() for c in a['caption']]
    return {'images': [a['image'] for a in annotations], 'texts': texts, 'img2txt': img2txt, 'wrongtext': wrongtext}


def mix_part(part, root):
    """Distractor images, one per COCO test image."""
    test = json.load(open(f'{root}/annotations/coco_karpathy_test.json'))
    return {'images': [f'{part}/{os.path.basename(a["image"])}' for a in test]}


def write_metadata(args):
    """meta.json per part (images, captions, index maps); returns all metadata."""
    metadata = {}
    for part in PARTS:
        path = part_path(args.features_dir, part, 'meta.json')
        if os.path.exists(path):
            metadata[part] = json.load(open(path))
            continue

        if part in ('train', 'val', 'test'):
            metadata[part] = karpathy_part(part, args.data_root)
        elif part.startswith('rococo_'):
            metadata[part] = rococo_part(part[len('rococo_'):], args.data_root)
        else:
            metadata[part] = mix_part(part, args.data_root)
        json.dump(metadata[part], open(path, 'w'))
        print(f'wrote {path}')
    return metadata


@torch.no_grad()
def images(args, metadata):
    """Raw CLIP features (fp16) of every part with images (all but the rococo_* caption sets)."""
    model, preprocess = clip_vitb32(args.device, f'{args.data_root}/.cache/clip')

    for part, meta in metadata.items():
        path = part_path(args.features_dir, part, 'image.pt')
        if part.startswith('rococo_') or os.path.exists(path):
            continue

        paths = [f'{args.image_root}/{x}' for x in meta['images']]
        loader = torch.utils.data.DataLoader(Images(paths, preprocess), batch_size=args.batch_size, num_workers=16)
        out = torch.cat([model.encode_image(x.to(args.device)).half().cpu() for x in loader])
        assert out.float().norm(dim=-1).min() > 0, 'zero embeddings (bad GPU?)'
        torch.save(out, path)
        print(f'wrote {path}')


@torch.no_grad()
def text(args, metadata):
    """Raw CLIP features (fp16) of every part with captions."""
    model, _ = clip_vitb32(args.device, f'{args.data_root}/.cache/clip')

    for part, meta in metadata.items():
        path = part_path(args.features_dir, part, 'text.pt')
        if 'texts' not in meta or os.path.exists(path):
            continue

        texts = meta['texts']
        out = torch.cat([model.encode_text(clip.tokenize(texts[i:i + args.batch_size], truncate=True).to(args.device)).half().cpu()
                         for i in range(0, len(texts), args.batch_size)])
        assert out.float().norm(dim=-1).min() > 0, 'zero embeddings (bad GPU?)'
        torch.save(out, path)
        print(f'wrote {path}')


@torch.no_grad()
def augmented_views(args, metadata):
    """k_aug image crop / flip / jitter views per train / val image, as clevr4 (the text view is another caption, sampled at train time)."""
    model, preprocess = clip_vitb32(args.device, f'{args.data_root}/.cache/clip')

    for split in ['train', 'val']:
        path = f'{args.features_dir}/{split}/image_aug{args.k_aug}.pt'
        if os.path.exists(path):
            continue

        set_seed(args.data_seed)
        paths = [f'{args.image_root}/{x}' for x in metadata[split]['images']]
        # batch size and workers also decide which worker augments which image, so changing them changes the views
        loader = torch.utils.data.DataLoader(Images(paths, lambda im: augment_image(im, preprocess, args.k_aug)),
                                             batch_size=args.batch_size, num_workers=16)
        out = [model.encode_image(x.flatten(0, 1).to(args.device)).half().cpu().view(len(x), args.k_aug, -1)
               for x in loader]
        torch.save(torch.cat(out).transpose(0, 1).contiguous(), path)
        print(f'wrote {path}')


@torch.no_grad()
def image_tokens(args, metadata):
    """ViT-B/32 patch tokens (fp16) of every part with images, and of the same k_aug views as augmented_views (train / val)."""
    model, preprocess = clip_vitb32(args.device, f'{args.data_root}/.cache/clip')

    for part, meta in metadata.items():
        path = part_path(args.features_dir, part, 'image_tokens.pt')
        if part.startswith('rococo_') or os.path.exists(path):
            continue

        paths = [f'{args.image_root}/{x}' for x in meta['images']]
        out = torch.empty(len(paths), *VITB32_IMAGE_TOKENS, dtype=torch.float16)
        i = 0
        loader = torch.utils.data.DataLoader(Images(paths, preprocess), batch_size=args.batch_size, num_workers=16)
        for x in loader:
            out[i:i + len(x)] = vitb32_image_tokens(model, x.to(args.device)).cpu()
            i += len(x)
        assert out[:, 0].float().norm(dim=-1).min() > 0, 'zero embeddings (bad GPU?)'
        torch.save(out, path)
        print(f'wrote {path}')

    for split in ['train', 'val']:
        path = f'{args.features_dir}/{split}/image_tokens_aug{args.k_aug}.pt'
        if os.path.exists(path):
            continue

        set_seed(args.data_seed)
        paths = [f'{args.image_root}/{x}' for x in metadata[split]['images']]
        out = torch.empty(args.k_aug, len(paths), *VITB32_IMAGE_TOKENS, dtype=torch.float16)
        i = 0
        loader = torch.utils.data.DataLoader(Images(paths, lambda im: augment_image(im, preprocess, args.k_aug)),
                                             batch_size=args.batch_size, num_workers=16)
        for x in loader:
            tokens = vitb32_image_tokens(model, x.flatten(0, 1).to(args.device)).view(len(x), args.k_aug, *VITB32_IMAGE_TOKENS)
            out[:, i:i + len(x)] = tokens.transpose(0, 1).cpu()
            i += len(x)
        torch.save(out, path)
        print(f'wrote {path}')


@torch.no_grad()
def text_tokens(args, metadata):
    """ViT-B/32 text tokens (fp16, cut to the part's longest caption) of every part with captions."""
    model, _ = clip_vitb32(args.device, f'{args.data_root}/.cache/clip')

    for part, meta in metadata.items():
        path = part_path(args.features_dir, part, 'text_tokens.pt')
        if 'texts' not in meta or os.path.exists(path):
            continue

        tokens = clip.tokenize(meta['texts'], truncate=True)
        length = int(tokens.argmax(-1).max()) + 1
        out = torch.empty(len(tokens), length, TEXT_TOKEN_DIM, dtype=torch.float16)
        for i in range(0, len(tokens), args.batch_size):
            out[i:i + args.batch_size] = vitb32_text_tokens(model, tokens[i:i + args.batch_size].to(args.device))[:, :length].cpu()
        assert out[:, 0].float().norm(dim=-1).min() > 0, 'zero embeddings (bad GPU?)'
        torch.save(out, path)
        print(f'wrote {path}')


@torch.no_grad()
def pre_projection(args, metadata):
    """Features before CLIP's projections (fp16) of every part, and CLIP's projections + logit scale (clip_projection.pt)."""
    model, preprocess = clip_vitb32(args.device, f'{args.data_root}/.cache/clip')

    path = f'{args.features_dir}/clip_projection.pt'
    if not os.path.exists(path):
        torch.save({'image': model.visual.proj.float().cpu(), 'text': model.text_projection.float().cpu(),
                    'logit_scale': model.logit_scale.float().cpu()}, path)
        print(f'wrote {path}')

    for part, meta in metadata.items():
        path = part_path(args.features_dir, part, 'image_pre_projection.pt')
        if part.startswith('rococo_') or os.path.exists(path):
            continue

        paths = [f'{args.image_root}/{x}' for x in meta['images']]
        loader = torch.utils.data.DataLoader(Images(paths, preprocess), batch_size=args.batch_size, num_workers=16)
        out = torch.cat([vitb32_image_pre_projection(model, x.to(args.device)).half().cpu() for x in loader])
        assert out.float().norm(dim=-1).min() > 0, 'zero embeddings (bad GPU?)'
        torch.save(out, path)
        print(f'wrote {path}')

    for part, meta in metadata.items():
        path = part_path(args.features_dir, part, 'text_pre_projection.pt')
        if 'texts' not in meta or os.path.exists(path):
            continue

        texts = meta['texts']
        out = torch.cat([vitb32_text_pre_projection(model, clip.tokenize(texts[i:i + args.batch_size], truncate=True).to(args.device)).half().cpu()
                         for i in range(0, len(texts), args.batch_size)])
        assert out.float().norm(dim=-1).min() > 0, 'zero embeddings (bad GPU?)'
        torch.save(out, path)
        print(f'wrote {path}')


def main():
    defaults = OmegaConf.load(f'{REPO}/configs/rococo.yaml')
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=defaults['data_root'])
    p.add_argument('--features_dir', default=defaults['features_dir'])
    p.add_argument('--k_aug', type=int, default=defaults['k_aug'])
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--data_seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--tokens', action='store_true', help='token features (RePercENT) instead of pooled features')
    p.add_argument('--pre_projection', action='store_true', help='features before CLIP\'s projections (finetuned_proj_clip) instead of pooled features')
    args = p.parse_args()
    args.image_root = f'{args.data_root}/MSCOCO/images'
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{args.features_dir}/{split}', exist_ok=True)

    metadata = write_metadata(args)
    if args.tokens:
        image_tokens(args, metadata)
        text_tokens(args, metadata)
        return
    if args.pre_projection:
        pre_projection(args, metadata)
        return

    images(args, metadata)
    text(args, metadata)
    augmented_views(args, metadata)


if __name__ == '__main__':
    main()
