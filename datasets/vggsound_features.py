"""
VGGSound: metadata, PE-AV (base, 16 frames) video / audio features per split, and k_aug augmented views (train / val);
with --tokens, their token features instead (RePercENT): video [16, 1024] frame tokens, audio [63, 1024] (4 tokens averaged).
Run: python -m datasets.vggsound_features [--tokens] (one GPU), or --part i --num_parts n in n processes (the last part to finish merges the parts).
"""
import argparse
import functools
import json
import os

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import tv_tensors
from torchvision.transforms import v2

from datasets.base import pe_av_base_16_frame
from utils import REPO

SPLITS = ['train', 'val', 'test']
NUM_FRAMES = 16
SAMPLE_RATE = 48000
AUDIO_TOKEN_POOL = 4   # audio tokens (25 per second) averaged per token feature
AUDIO_TOKENS = 63      # token features per clip: 10 s, one per 160 ms


def read_frames(path):
    """NUM_FRAMES evenly spaced RGB frames [n, h, w, 3], picked as in the PE-AV video processor."""
    with av.open(path) as f:
        frames = list(f.decode(video=0))
    idx = [int(i * (len(frames) - 1) / (NUM_FRAMES - 1)) for i in range(NUM_FRAMES)]
    return np.stack([frames[i].to_ndarray(format='rgb24') for i in idx])


def read_audio(path):
    """Mono waveform resampled to SAMPLE_RATE."""
    with av.open(path) as f:
        resampler = av.AudioResampler(format='flt', layout='mono', rate=SAMPLE_RATE)
        chunks = [r.to_ndarray() for frame in f.decode(audio=0) for r in resampler.resample(frame)]
        chunks += [r.to_ndarray() for r in resampler.resample(None)]
    return np.concatenate(chunks, axis=1)[0]


def augment_video(frames, seed):
    """One per view, same transform on every frame: random resized crop, horizontal flip, color jitter or slight blur."""
    torch.manual_seed(seed)
    augmentation = v2.RandomChoice([
        v2.RandomResizedCrop(336, scale=(0.8, 1.0), ratio=(0.75, 1.33), antialias=True),
        v2.Compose([v2.Resize((336, 336), antialias=True), v2.RandomHorizontalFlip(p=1.0)]),
        v2.Compose([v2.Resize((336, 336), antialias=True), v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)]),
        v2.Compose([v2.Resize((336, 336), antialias=True), v2.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5))]),
    ])
    video = augmentation(tv_tensors.Video(torch.from_numpy(frames).permute(0, 3, 1, 2)))
    return video.permute(0, 2, 3, 1).contiguous().numpy()


def augment_audio(audio, seed):
    """One per view: gain within +-3 dB, or one zeroed stretch of at most 0.25 s."""
    rng = np.random.default_rng(seed)
    if rng.random() < 0.5:
        return audio * 10 ** (rng.uniform(-3, 3) / 20)

    audio = audio.copy()
    width = int(rng.uniform(0, 0.25) * SAMPLE_RATE)
    start = int(rng.uniform(0, max(1, len(audio) - width)))
    audio[start:start + width] = 0
    return audio


def process_batch(processor, k_aug, data_seed, items):
    """Collate in the worker: decode each (index, video path, audio path) once, drop clips that cannot be decoded, run the PE-AV processor
    on the clip itself (k_aug=0) or on k_aug views seeded by (data_seed, index, view)."""
    kept, videos, audios = [], [], []
    for i, video_path, audio_path in items:
        try:
            frames, audio = read_frames(video_path), read_audio(audio_path)
        except Exception:
            continue
        kept.append(i)
        if k_aug == 0:
            videos.append(frames)
            audios.append(audio)
        for view in range(k_aug):
            seed = int(np.random.SeedSequence([data_seed, i, view]).generate_state(1)[0])
            videos.append(augment_video(frames, seed))
            audios.append(augment_audio(audio, seed))
    if not kept:
        return [], None

    inputs = processor(videos=videos, audio=audios, sampling_rate=SAMPLE_RATE, do_sample_frames=False,
                       padding=True, return_tensors='pt')
    return kept, inputs


def write_metadata(args):
    """meta.json per split (clip ids, labels); val is a val_frac of the train YouTube videos."""
    paths = {split: f'{args.features_dir}/{split}/meta.json' for split in SPLITS}
    if all(os.path.exists(p) for p in paths.values()):
        return {split: json.load(open(p)) for split, p in paths.items()}

    clips = pd.read_csv(f'{args.data_root}/vggsound.csv', header=None, names=['youtube_id', 'start', 'label', 'split'])
    clips['id'] = clips['youtube_id'] + '_' + clips['start'].map('{:06d}'.format)
    downloaded = {os.path.splitext(f)[0] for f in os.listdir(args.video_root)}
    clips = clips[clips['id'].isin(downloaded)]

    train = clips[clips['split'] == 'train']
    videos = train['youtube_id'].drop_duplicates().sample(frac=1, random_state=args.data_seed)
    val_videos = set(videos[:round(len(videos) * args.val_frac)])
    is_val = train['youtube_id'].isin(val_videos)
    splits = {'train': train[~is_val], 'val': train[is_val], 'test': clips[clips['split'] == 'test']}

    metadata = {}
    for split, rows in splits.items():
        rows = rows.sort_values('id')
        metadata[split] = {'ids': rows['id'].tolist(), 'labels': rows['label'].tolist()}
        json.dump(metadata[split], open(f'{paths[split]}.{os.getpid()}.tmp', 'w'))
        os.replace(f'{paths[split]}.{os.getpid()}.tmp', paths[split])
    return metadata


@torch.no_grad()
def embed(args, metadata, k_aug):
    """PE-AV video / audio embeddings (fp16) [n, views, dim] (args.tokens: [n, views, tokens, dim]) and the decoded clip ids
    of part args.part of args.num_parts: of the clips themselves (k_aug=0, every split) or of k_aug augmented views per clip (train / val)."""
    model, processor = pe_av_base_16_frame(args.device, f'{args.data_root}/.cache/pe_av')
    suffix = ('_tokens' if args.tokens else '') + (f'_aug{k_aug}' if k_aug else '')

    for split in (['train', 'val'] if k_aug else SPLITS):
        meta = metadata[split]
        paths = {m: f'{args.features_dir}/{split}/parts/{m}{suffix}_{args.part}_of_{args.num_parts}.pt' for m in ['video', 'audio', 'kept']}
        merged = [f'{args.features_dir}/{split}/{m}{suffix}.pt' for m in ['video', 'audio']]
        if all(os.path.exists(p) for p in paths.values()) or all(os.path.exists(p) for p in merged):
            continue

        rows = np.array_split(np.arange(len(meta['ids'])), args.num_parts)[args.part].tolist()
        clips = [(i, f'{args.video_root}/{meta["ids"][i]}.mp4', f'{args.video_root}/{meta["ids"][i]}.mp4') for i in rows]
        loader = torch.utils.data.DataLoader(clips, batch_size=args.batch_size // max(1, k_aug), num_workers=16,
                                             collate_fn=functools.partial(process_batch, processor, k_aug, args.data_seed), timeout=600)
        kept, video, audio = [], [], []
        for idx, inputs in loader:
            if inputs is None:
                continue

            inputs = inputs.to(args.device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if args.tokens:
                    output = model.audio_video_encoder(**inputs)
                else:
                    output = model(**inputs)

            if args.tokens:
                frames = output.video_model_output.last_hidden_state
                length = AUDIO_TOKENS * AUDIO_TOKEN_POOL
                mask = output.audio_model_output.output_mask.float()[:, :length]
                tokens = output.audio_model_output.last_hidden_state.float()[:, :mask.shape[1]] * mask.unsqueeze(-1)
                tokens, mask = F.pad(tokens, (0, 0, 0, length - mask.shape[1])), F.pad(mask, (0, length - mask.shape[1]))
                groups = (AUDIO_TOKENS, AUDIO_TOKEN_POOL)
                sounds = tokens.unflatten(1, groups).sum(2) / mask.unflatten(1, groups).sum(2, keepdim=True).clamp(min=1)
                video.append(frames.half().cpu().unflatten(0, (len(idx), -1)))
                audio.append(sounds.half().cpu().unflatten(0, (len(idx), -1)))
            else:
                video.append(output.video_embeds.half().cpu().unflatten(0, (len(idx), -1)))
                audio.append(output.audio_embeds.half().cpu().unflatten(0, (len(idx), -1)))
            kept += [meta['ids'][i] for i in idx]

        os.makedirs(f'{args.features_dir}/{split}/parts', exist_ok=True)
        for name, x in [('video', torch.cat(video)), ('audio', torch.cat(audio)), ('kept', kept)]:
            torch.save(x, f'{paths[name]}.{os.getpid()}.tmp')
            os.replace(f'{paths[name]}.{os.getpid()}.tmp', paths[name])
        print(f'wrote {split}{suffix} part {args.part} of {args.num_parts}: {len(kept)} of {len(clips)} clips')


def merge(args, k_aug):
    """
    Once every part of a split is written, write them into {video,audio}.pt [n, dim] (k_aug=0; clips that could not be
    decoded are dropped from meta.json, pooled features only) or {video,audio}_aug{k_aug}.pt [k_aug, n, dim], rows as in meta.json
    (args.tokens: *_tokens*.pt with a tokens axis before dim).
    """
    suffix = ('_tokens' if args.tokens else '') + (f'_aug{k_aug}' if k_aug else '')
    for split in (['train', 'val'] if k_aug else SPLITS):
        paths = {m: f'{args.features_dir}/{split}/{m}{suffix}.pt' for m in ['video', 'audio']}
        part_paths = {m: [f'{args.features_dir}/{split}/parts/{m}{suffix}_{i}_of_{args.num_parts}.pt' for i in range(args.num_parts)]
                      for m in ['video', 'audio', 'kept']}
        if all(os.path.exists(p) for p in paths.values()) or not all(os.path.exists(p) for ps in part_paths.values() for p in ps):
            continue

        kept_parts = [torch.load(p) for p in part_paths['kept']]
        kept = [clip for part in kept_parts for clip in part]
        meta_path = f'{args.features_dir}/{split}/meta.json'
        meta = json.load(open(meta_path))
        if not k_aug and not args.tokens:
            labels = dict(zip(meta['ids'], meta['labels']))
            meta = {'ids': kept, 'labels': [labels[clip] for clip in kept]}
            json.dump(meta, open(f'{meta_path}.{os.getpid()}.tmp', 'w'))
            os.replace(f'{meta_path}.{os.getpid()}.tmp', meta_path)

        missing = set(meta['ids']) - set(kept)
        assert not missing, f'{len(missing)} clips of {meta_path} missing from the parts'
        position = {clip: r for r, clip in enumerate(meta['ids'])}
        for m, path in paths.items():
            x = None
            for part_path, part_kept in zip(part_paths[m], kept_parts):
                part = torch.load(part_path)
                rows = [position[clip] for clip in part_kept]
                if x is None:
                    shape = (k_aug, len(position)) if k_aug else (len(position),)
                    x = torch.empty(shape + part.shape[2:], dtype=part.dtype)
                if k_aug:
                    x[:, rows] = part.transpose(0, 1)
                else:
                    x[rows] = part.squeeze(1)
            torch.save(x, f'{path}.{os.getpid()}.tmp')
            os.replace(f'{path}.{os.getpid()}.tmp', path)
        print(f'wrote {paths["video"]}, {paths["audio"]}: {len(position)} clips')


def main():
    defaults = OmegaConf.load(f'{REPO}/configs/vggsound.yaml')
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=defaults['data_root'])
    p.add_argument('--features_dir', default=defaults['features_dir'])
    p.add_argument('--val_frac', type=float, default=defaults['val_frac'])
    p.add_argument('--k_aug', type=int, default=defaults['k_aug'])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--data_seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--part', type=int, default=0)
    p.add_argument('--num_parts', type=int, default=1)
    p.add_argument('--tokens', action='store_true', help='token features (RePercENT) instead of pooled features')
    args = p.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.video_root = f'{args.data_root}/video'
    for split in SPLITS:
        os.makedirs(f'{args.features_dir}/{split}', exist_ok=True)

    metadata = write_metadata(args)
    for k_aug in [0, args.k_aug]:
        embed(args, metadata, k_aug)
        merge(args, k_aug)


if __name__ == '__main__':
    main()
