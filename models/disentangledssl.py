import torch

from utils import THIRD_PARTY, import_upstream
from .base import Method, train_loop
from .fanchor import Adapter

algorithms = import_upstream(f'{THIRD_PARTY}/DisentangledSSL/synthetic_task', 'algorithms')


class DisentangledSSL(Method):
    """
    Two-stage upstream DisentangledSSL: step1 (MVInfoMax) learns shared s1, s2; then frozen,
    step2 (DisenModel) learns private p1, p2 using step1's shared codes and augmented views.
    """
    uses_views = True   # batches are (x1, x2, v1, v2), v = augmented views

    def loss(self, views):
        # step1 uses (x1, x2), step2 all four
        loss = self.model(*views[:self.n_inputs])[0]
        return loss, {'loss': loss.item()}

    def fit_stage(self, model, data, epochs, n_inputs, name):
        """Train one stage with its own Adam + cosine schedule."""
        self.model, self.n_inputs = model, n_inputs
        self.optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=0)

        return train_loop(model, data, epochs, self.train_step, self.val_step, views=True,
                          record_every=self.args.record_every, end_epoch=self.scheduler.step, name=name)

    def fit(self, data):
        args = self.args
        self.step1 = algorithms.MVInfoMaxModel(
            *data.dims, args.hidden_dim, args.embed_dim, layers=args.layers, initialization='normal',
            distribution='vmf', vmfkappa=args.kappa,
            beta_start_value=args.beta, beta_end_value=args.beta)
        if args.shared_residual:
            self.step1.encoder_x1 = Adapter(data.dims[0], args.embed_dim, args.hidden_dim, args.layers, 'relu')
            self.step1.encoder_x2 = Adapter(data.dims[1], args.embed_dim, args.hidden_dim, args.layers, 'relu')
        self.step1.to(args.dev)
        hist_s1 = self.fit_stage(self.step1, data, args.epochs_s1, n_inputs=2, name='step1')
        self.step1.requires_grad_(False)   # step1 frozen during step2

        self.step2 = algorithms.DisenModel(
            self.step1, *data.dims, args.hidden_dim, args.private_dim, zs_dim=args.embed_dim, layers=args.layers,
            initialization='normal', lmd_start_value=args.lmd, lmd_end_value=args.lmd,
            usezsx=False).to(args.dev)
        hist_s2 = self.fit_stage(self.step2, data, args.epochs_s2, n_inputs=4, name='step2')

        # final losses of both stages, prefixed step1_ / step2_
        losses = {f'step1_{k}': v for k, v in hist_s1[-1].items() if k != 'epoch'}
        losses.update({f'step2_{k}': v for k, v in hist_s2[-1].items() if k != 'epoch'})

        return {'final_losses': losses, 'hist_s1': hist_s1, 'hist_s2': hist_s2}

    @torch.no_grad()
    def embed(self, x1, x2):
        """Shared from step1, private from step2."""
        self.step1.eval()
        self.step2.eval()

        s1, s2 = self.step1.get_separate_embeddings([x1, x2])
        p1, p2 = self.step2.get_embedding([x1, x2])
        return s1, p1, s2, p2

    @torch.no_grad()
    def embed_shared(self, x, modality):
        """step1 encoder output, used for retrieval."""
        self.step1.eval()
        encoder = self.step1.encoder_x1 if modality == 0 else self.step1.encoder_x2
        return encoder(x)
