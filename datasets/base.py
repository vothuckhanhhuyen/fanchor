import torch
from PIL import Image


class Dataset:
    """
    Features on the GPU (token features on the CPU, see load_features). args.data_root: raw data + labels; args.features_dir: everything extracted from it.
    Subclasses set:
      x[split] = (image, text) features, split in train / val (/ test)
      x_aug[split] = (image, text) augmented views, each [k_aug, n, dim], only when views=True
      dims = (image dim, text dim), the last axis (token features: [n, tokens, dim])
    """
    sequences = False   # True if inputs are sequences: methods then use data.encoders instead of MLPs

    def __init__(self, args, views=False):
        self.args = args

    def features(self, split, modality):
        return self.x[split][0 if modality == 'image' else 1]

    def batches(self, split, shuffle, views=False):
        """Yields (x1, x2, ...), or (x1, x2, v1, v2) with one random augmented view per sample, on args.dev; train drops the last partial batch."""
        x = self.x[split]
        idx = torch.arange(len(x[0]), device=x[0].device)
        for b in index_batches(idx, self.args.batch_size, shuffle, drop_last=split == 'train'):
            if views:
                k = torch.randint(self.args.k_aug, (len(b),), device=b.device)
                batch = x[0][b], x[1][b], self.x_aug[split][0][k, b], self.x_aug[split][1][k, b]
            else:
                batch = tuple(v[b] for v in x)
            yield tuple(t.to(self.args.dev) for t in batch)


def load_features(path, args):
    """Pooled features: float on args.dev. Token features (args.tokens): fp16, memory-mapped on the CPU, batches moved to args.dev."""
    return torch.load(path, mmap=True) if args.tokens else torch.load(path).float().to(args.dev)


def index_batches(idx, batch_size, shuffle, drop_last):
    """Split idx into batches; drop_last drops the final partial batch (train, as in OpenCLIP)."""
    if shuffle:
        idx = idx[torch.randperm(len(idx), device=idx.device)]
    if drop_last:
        idx = idx[:len(idx) // batch_size * batch_size]
    return idx.split(batch_size)


class Images(torch.utils.data.Dataset):
    """Image files -> transform(image); used by the *_features.py extraction scripts."""
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert('RGB'))


def clip_rn50(dev):
    """OpenAI CLIP RN50 via open_clip (clevr4): returns model, preprocess, tokenizer."""
    import open_clip   # imported here so training never loads it
    clip, _, preprocess = open_clip.create_model_and_transforms('RN50-quickgelu', pretrained='openai')
    return clip.to(dev).eval(), preprocess, open_clip.get_tokenizer('RN50-quickgelu')


def clip_vitb32(dev, download_root):
    """OpenAI CLIP ViT-B/32 via the clip package (rococo): returns model, preprocess."""
    import clip   # imported here so training never loads it
    model, preprocess = clip.load('ViT-B/32', device=dev, download_root=download_root)
    return model.eval(), preprocess


RN50_IMAGE_TOKENS = (49, 2048)     # token feature shapes per sample
VITB32_IMAGE_TOKENS = (49, 768)
TEXT_TOKEN_DIM = 512                # text: [longest caption in tokens, 512]


def rn50_image_tokens(model, x):
    """open_clip RN50: the 7x7 grid the attention pool reads (layer4 output + its position embedding), [n, 49, 2048]."""
    visual = model.visual
    x = visual.layer4(visual.layer3(visual.layer2(visual.layer1(visual.stem(x)))))
    return x.flatten(2).transpose(1, 2) + visual.attnpool.positional_embedding[1:].to(x.dtype)


def rn50_text_tokens(model, tokens):
    """open_clip RN50 text tokens after ln_final (before the projection), zero after the EOT token, [n, 77, 512]."""
    x = model.token_embedding(tokens) + model.positional_embedding
    x = model.ln_final(model.transformer(x, attn_mask=model.attn_mask))
    return x * (torch.arange(tokens.shape[1], device=tokens.device) <= tokens.argmax(-1, keepdim=True)).unsqueeze(-1)


def vitb32_image_tokens(model, x):
    """clip ViT-B/32 patch tokens after ln_post (before the projection; RePercENT's IRFL inputs), [n, 49, 768]."""
    visual = model.visual
    x = visual.conv1(x.type(model.dtype)).flatten(2).transpose(1, 2)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(len(x), 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = visual.ln_pre(torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype))
    return visual.ln_post(visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)[:, 1:])


def vitb32_text_tokens(model, tokens):
    """clip ViT-B/32 text tokens after ln_final (before the projection), zero after the EOT token, [n, 77, 512]."""
    x = model.token_embedding(tokens).type(model.dtype) + model.positional_embedding.type(model.dtype)
    x = model.ln_final(model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2))
    return x * (torch.arange(tokens.shape[1], device=tokens.device) <= tokens.argmax(-1, keepdim=True)).unsqueeze(-1)


def vitb32_image_pre_projection(model, x):
    """clip ViT-B/32 CLS token after ln_post (encode_image before visual.proj; rococo), [n, 768]."""
    visual = model.visual
    x = visual.conv1(x.type(model.dtype)).flatten(2).transpose(1, 2)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(len(x), 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = visual.ln_pre(torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype))
    return visual.ln_post(visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)[:, 0])


def vitb32_text_pre_projection(model, tokens):
    """clip ViT-B/32 EOT token after ln_final (encode_text before text_projection; rococo), [n, 512]."""
    x = model.token_embedding(tokens).type(model.dtype) + model.positional_embedding.type(model.dtype)
    x = model.ln_final(model.transformer(x.permute(1, 0, 2)).permute(1, 0, 2))
    return x[torch.arange(len(x)), tokens.argmax(-1)]


def pe_av_base_16_frame(dev, cache_dir):
    """PE-AV base, 16 frames per video, via transformers (vggsound): returns model, processor."""
    from transformers import PeAudioVideoModel, PeAudioVideoProcessor   # imported here so training never loads it
    name = 'facebook/pe-av-base-16-frame'
    model = PeAudioVideoModel.from_pretrained(name, cache_dir=cache_dir)
    return model.to(dev).eval(), PeAudioVideoProcessor.from_pretrained(name, cache_dir=cache_dir)
