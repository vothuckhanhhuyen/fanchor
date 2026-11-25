import torch

from .base import Method
from .fanchor import Adapter
from .disentangledssl import algorithms


class JointDisen(Method):
    """
    Upstream DisentangledSSL, joint version: shared and private learned together in one stage,
    max I(zs;y) + a * [I(zs,zp;X) - lmd * I(zs;zp)], with the augmented view as y.
    """
    uses_views = True   # batches are (x1, x2, v1, v2), v = augmented views

    def __init__(self, args, data):
        super().__init__(args, data)
        self.model = algorithms.JointDisenModel(
            *data.dims, args.hidden_dim, args.embed_dim, layers=args.layers,
            initialization='normal', distribution='vmf', vmfkappa=args.kappa,
            lmd_start_value=args.lmd, lmd_end_value=args.lmd,
            a=args.a, usezsx=False)
        if args.private_dim != args.embed_dim:
            self.model.encoder_x1 = algorithms.mlp(data.dims[0] + args.embed_dim, args.hidden_dim, args.private_dim, args.layers, 'relu', initialization='normal')
            self.model.encoder_x2 = algorithms.mlp(data.dims[1] + args.embed_dim, args.hidden_dim, args.private_dim, args.layers, 'relu', initialization='normal')
        if args.shared_residual:
            self.model.encoder_x1s = Adapter(data.dims[0], args.embed_dim, args.hidden_dim, args.layers, 'relu')
            self.model.encoder_x2s = Adapter(data.dims[1], args.embed_dim, args.hidden_dim, args.layers, 'relu')
        self.model.to(args.dev)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs, eta_min=0)

    def loss(self, views):
        loss = self.model(*views)[0]
        return loss, {'loss': loss.item()}

    @torch.no_grad()
    def embed(self, x1, x2):
        # upstream returns (s1, s2, p1, p2); reorder to (s1, p1, s2, p2)
        self.model.eval()
        s1, s2, p1, p2 = self.model.get_embedding([x1, x2])
        return s1, p1, s2, p2

    @torch.no_grad()
    def embed_shared(self, x, modality):
        """Shared encoder output, used for retrieval."""
        self.model.eval()
        encoder = self.model.encoder_x1s if modality == 0 else self.model.encoder_x2s
        return encoder(x)
