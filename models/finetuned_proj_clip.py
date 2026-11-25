import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Method, infonce_loss


class FinetunedProjCLIPModel(nn.Module):
    """CLIP's image / text projections (initialized from CLIP) on frozen pre-projection features, trained with CLIP's symmetric InfoNCE."""
    def __init__(self, projections, learn_tau=True):
        super().__init__()
        self.projs = nn.ParameterList([nn.Parameter(projections['image'].clone()), nn.Parameter(projections['text'].clone())])

        # tau = 1 / exp(logit_scale), capped at tau >= 0.01 (as CLIP)
        self.logit_scale = nn.Parameter(projections['logit_scale'].clone(), requires_grad=learn_tau)
        self._max_logit_scale = torch.log(torch.tensor(100.0)).item()

    def get_embedding(self, xs):
        return [x @ proj for x, proj in zip(xs, self.projs)]

    def forward(self, xs):
        with torch.no_grad():
            self.logit_scale.clamp_(max=self._max_logit_scale)
        tau = 1.0 / self.logit_scale.exp()

        z1, z2 = [F.normalize(z, dim=-1) for z in self.get_embedding(xs)]
        loss_align, loss_uniform = infonce_loss(z1, z2, tau)
        loss = loss_align + loss_uniform

        return loss, {'loss': loss.item(), 'align': loss_align.item(), 'uniform': loss_uniform.item(), 'tau': tau.item()}


class FinetunedProjCLIP(Method):
    """Finetune only CLIP's projections (args.pre_projection features; clip_projection.pt from datasets.rococo_features --pre_projection)."""

    def __init__(self, args, data):
        super().__init__(args, data)
        projections = torch.load(f'{args.features_dir}/clip_projection.pt')
        self.model = FinetunedProjCLIPModel(projections).to(args.dev)

        # as OpenCLIP finetuning: no weight decay on logit_scale, AdamW betas (0.9, 0.98), eps 1e-6, linear warmup then per-step cosine to 0
        groups = [{'params': list(self.model.projs), 'weight_decay': args.weight_decay},
                  {'params': [p for p in [self.model.logit_scale] if p.requires_grad], 'weight_decay': 0.0}]
        self.optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6)

        steps_per_epoch = len(data.x['train'][0]) // args.batch_size
        warmup, total = args.warmup_epochs * steps_per_epoch, args.epochs * steps_per_epoch
        self.lr_schedule = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))

    def loss(self, xs):
        return self.model(xs)

    def train_step(self, batch):
        logs = super().train_step(batch)
        self.lr_schedule.step()
        return logs

    @torch.no_grad()
    def embed(self, *xs):
        """s = projection outputs, p = the pre-projection inputs (only used for cka)."""
        self.model.eval()
        z1, z2 = self.model.get_embedding(xs)
        return z1, xs[0], z2, xs[1]

    @torch.no_grad()
    def embed_shared(self, x, modality):
        self.model.eval()
        return x @ self.model.projs[modality]
