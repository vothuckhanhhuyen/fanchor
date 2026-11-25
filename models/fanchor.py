import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import Method, mlp, multimodal_contrastive_loss, train_loop


class Adapter(nn.Module):
    """MLP on frozen features; if residual, shortcut + MLP with the MLP's last layer zero-init (starts as the shortcut)."""
    def __init__(self, in_dim, out_dim, hidden, num_layers=4, activation='relu', residual=True):
        super().__init__()
        self.residual = residual
        self.net = mlp(in_dim, hidden, out_dim, num_layers, activation)

        if residual:
            self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        if self.residual:
            return self.shortcut(x) + self.net(x)
        return self.net(x)


class FAnchorModel(nn.Module):
    """
    One shared adapter (s_m) and one private adapter (p_m) per modality, embed_dims = input dim per modality (>= 2 modalities);
    shared adapters trained with the contrastive loss;
    private adapters trained by reconstructing each input from the detached shared anchor + its private code.
    hier_cond: private also sees the anchor.
    stop_grad: detach the anchor; if False, recon (and hier_cond) gradients also reach the shared adapters.
    share_decoder: decoders share all layers across modalities except the last.
    """
    def __init__(self, embed_dims=(256, 256),
                 shared_dim=None, shared_hidden=None, shared_layers=3, shared_residual=False,
                 private_dim=None, private_hidden=None, private_layers=3, private_residual=False,
                 hier_cond=False, stop_grad=True,
                 decoder_hidden=None, decoder_layers=3, share_decoder=False,
                 lam=1.0,
                 recon_loss_type='recon', cl_loss_type='fanchor',
                 activation='relu', learn_tau=True, tau=0.07):
        super(FAnchorModel, self).__init__()
        self.lam = lam
        self.hier_cond = hier_cond
        self.stop_grad = stop_grad

        if recon_loss_type not in ('additive', 'additive_matryoshka', 'concat', 'concat_matryoshka'):
            raise ValueError(f"Unknown recon_loss_type: {recon_loss_type!r}")
        self.recon_loss_type = recon_loss_type

        if cl_loss_type not in ('fanchor', 'infonce', 'centrobind'):
            raise ValueError(f"Unknown cl_loss_type: {cl_loss_type!r}")
        self.cl_loss_type = cl_loss_type

        shared_dim = max(embed_dims) // 2 if shared_dim is None else shared_dim
        private_dim = max(embed_dims) // 2 if private_dim is None else private_dim

        shared_hidden = shared_dim if shared_hidden is None else shared_hidden
        private_hidden = private_dim if private_hidden is None else private_hidden
        decoder_hidden = max(embed_dims) if decoder_hidden is None else decoder_hidden

        self.shared_adapters = nn.ModuleList(Adapter(embed_dim, shared_dim, hidden=shared_hidden, num_layers=shared_layers, activation=activation, residual=shared_residual)
                                             for embed_dim in embed_dims)

        self.private_adapters = nn.ModuleList(Adapter(embed_dim + shared_dim if hier_cond else embed_dim, private_dim, hidden=private_hidden, num_layers=private_layers, activation=activation, residual=private_residual)
                                              for embed_dim in embed_dims)

        # concat: one decoder on [anchor, private]; additive: separate anchor / private decoders, outputs summed
        if recon_loss_type in ('concat', 'concat_matryoshka'):
            self.decoders = nn.ModuleList(mlp(shared_dim + private_dim, decoder_hidden, embed_dim, decoder_layers, activation)
                                          for embed_dim in embed_dims)
        else:
            self.decoders_a = nn.ModuleList(mlp(shared_dim, decoder_hidden, embed_dim, decoder_layers, activation)
                                            for embed_dim in embed_dims)
            self.decoders_p = nn.ModuleList(mlp(private_dim, decoder_hidden, embed_dim, decoder_layers, activation)
                                            for embed_dim in embed_dims)

        # share_decoder: every modality reuses modality 0's decoder layers except its own last layer
        if share_decoder:
            for name in ('decoders', 'decoders_a', 'decoders_p'):
                if hasattr(self, name):
                    decoders = getattr(self, name)
                    setattr(self, name, nn.ModuleList(nn.Sequential(*decoders[0][:-1], decoder[-1]) for decoder in decoders))

        # tau = 1 / exp(logit_scale), capped at tau >= 0.01
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / tau))), requires_grad=learn_tau)
        self._max_logit_scale = float(np.log(1.0 / 0.01))

    def get_embedding_from_z(self, *es):
        """One input per modality -> (s1, p1, s2, p2, ...)."""
        ss = [adapter(e) for adapter, e in zip(self.shared_adapters, es)]

        if self.hier_cond:
            anchor = F.normalize(sum(F.normalize(s, dim=-1) for s in ss) / len(ss), dim=-1)
            if self.stop_grad:
                anchor = anchor.detach()
            p_ins = [torch.cat([e, anchor], dim=-1) for e in es]
        else:
            p_ins = es

        ps = [adapter(p_in) for adapter, p_in in zip(self.private_adapters, p_ins)]

        return tuple(t for s, p in zip(ss, ps) for t in (s, p))

    def decode(self, anchor, ps):
        """
        Reconstructions of each input: full, from anchor alone (recon_a), from private alone (recon_p).
          additive: full = dec_a(anchor) + dec_p(private), recon_a = dec_a(anchor), recon_p = dec_p(private)
          concat:   full = D([anchor, private]), recon_a = D([anchor, 0]), recon_p = D([0, private])
        """
        if self.recon_loss_type in ('concat', 'concat_matryoshka'):
            zeros_a, zeros_p = torch.zeros_like(anchor), torch.zeros_like(ps[0])
            full = [decoder(torch.cat([anchor, p], dim=-1)) for decoder, p in zip(self.decoders, ps)]
            from_anchor = [decoder(torch.cat([anchor, zeros_p], dim=-1)) for decoder in self.decoders]
            from_private = [decoder(torch.cat([zeros_a, p], dim=-1)) for decoder, p in zip(self.decoders, ps)]
            return full, from_anchor, from_private

        from_anchor = [decoder(anchor) for decoder in self.decoders_a]
        from_private = [decoder(p) for decoder, p in zip(self.decoders_p, ps)]
        full = [a + p for a, p in zip(from_anchor, from_private)]
        return full, from_anchor, from_private

    def forward(self, *es):
        with torch.no_grad():
            self.logit_scale.clamp_(max=self._max_logit_scale)
        tau = 1.0 / self.logit_scale.exp()

        # shared: contrastive loss; anchor detached so recon gradients never reach the shared adapters
        blocks = [F.normalize(t, dim=-1) for t in self.get_embedding_from_z(*es)]
        ss, ps = blocks[0::2], blocks[1::2]
        loss_align, loss_uniform = multimodal_contrastive_loss(ss, tau, self.cl_loss_type)
        anchor = F.normalize(sum(ss) / len(ss), dim=-1)
        if self.stop_grad:
            anchor = anchor.detach()

        # recon target: L2-normalized inputs
        targets = tuple(F.normalize(t, dim=-1) for t in es)

        full, from_anchor, from_private = self.decode(anchor, ps)
        loss_full = reconstruction_mse(full, targets)
        loss_recon_a = reconstruction_mse(from_anchor, targets)
        loss_recon_p = reconstruction_mse(from_private, targets)

        # matryoshka: anchor alone must also reconstruct; recon_p (private alone) is logged only
        if self.recon_loss_type in ('concat_matryoshka', 'additive_matryoshka'):
            loss_recon = loss_recon_a + loss_full
        else:
            loss_recon = loss_full

        loss = loss_align + self.lam * loss_uniform + loss_recon
        return loss, {'loss': loss.item(), 'align': loss_align.item(), 'uniform': loss_uniform.item(),
                      'recon': loss_recon.item(), 'recon_a': loss_recon_a.item(),
                      'recon_p': loss_recon_p.item(), 'tau': tau.item()}

    def build_optimizer(self, lr_shared=1e-3, lr_private=1e-4):
        """Adam with separate lr for shared (+ tau) and private + decoders."""
        shared = [self.shared_adapters]
        private = [m for name, m in self.named_children() if name.startswith(('private_adapters', 'decoders'))]

        return optim.Adam([
            {'params': [p for m in shared for p in m.parameters()] + [self.logit_scale], 'lr': lr_shared},
            {'params': [p for m in private for p in m.parameters()], 'lr': lr_private},
        ])


def reconstruction_mse(recons, targets):
    return sum(((r - t) ** 2).sum(-1).mean() for r, t in zip(recons, targets)) / len(recons)


class FAnchor(Method):
    """Method wrapper: builds the model from args on the dataset's frozen features."""
    has_anchor = True

    def __init__(self, args, data):
        super().__init__(args, data)
        self.model = FAnchorModel(
            embed_dims=data.dims,
            shared_dim=args.shared_dim, shared_hidden=args.shared_hidden, shared_layers=args.shared_layers,
            shared_residual=args.shared_residual,
            private_dim=args.private_dim, private_hidden=args.private_hidden, private_layers=args.private_layers,
            decoder_hidden=args.decoder_hidden, decoder_layers=args.decoder_layers, share_decoder=args.share_decoder,
            lam=args.lam, hier_cond=args.hier_cond, stop_grad=args.stop_grad,
            recon_loss_type=args.recon_loss_type, cl_loss_type=args.cl_loss_type,
        ).to(args.dev)

        self.optimizer = self.model.build_optimizer(lr_shared=args.lr_shared, lr_private=args.lr_private)

    def loss(self, batch):
        return self.model(*batch)

    def fit(self, data):
        if not self.args.two_stage:
            return super().fit(data)

        # stage 1: shared adapters + tau only; stage 2: shared frozen, private adapters + decoders only
        args, model = self.args, self.model
        shared = list(model.shared_adapters.parameters()) + [model.logit_scale]
        private = [p for name, m in model.named_children() if name.startswith(('private_adapters', 'decoders')) for p in m.parameters()]

        self.optimizer = optim.Adam(shared, lr=args.lr_shared)
        hist_s1 = train_loop(model, data, args.epochs, self.train_step, self.val_step, record_every=args.record_every, name='stage1')

        for p in shared:
            p.requires_grad_(False)
        self.optimizer = optim.Adam(private, lr=args.lr_private)
        hist_s2 = train_loop(model, data, args.epochs, self.train_step, self.val_step, record_every=args.record_every, name='stage2')

        losses = {f'stage1_{k}': v for k, v in hist_s1[-1].items() if k != 'epoch'}
        losses.update({f'stage2_{k}': v for k, v in hist_s2[-1].items() if k != 'epoch'})

        return {'final_losses': losses, 'hist_s1': hist_s1, 'hist_s2': hist_s2}

    @torch.no_grad()
    def embed(self, *xs):
        self.model.eval()
        return self.model.get_embedding_from_z(*xs)

    @torch.no_grad()
    def embed_shared(self, x, modality):
        self.model.eval()
        return self.model.shared_adapters[modality](x)
