import torch

from utils import THIRD_PARTY, import_upstream
from .base import Method
from .fanchor import Adapter

FactorCLSSL = import_upstream(f'{THIRD_PARTY}/FactorCL', 'multibench_model').FactorCLSSL


class FactorCL(Method):
    """Upstream FactorCL-SSL on our MLP encoders; the augmented view plays the role of the label y."""
    uses_views = True   # batches are (x1, x2, v1, v2), v = augmented views

    def __init__(self, args, data):
        super().__init__(args, data)
        if data.sequences:
            encoders = list(data.encoders(args.embed_dim))
        else:
            encoders = [Adapter(d, args.embed_dim, args.hidden_dim, args.layers, 'relu', residual=args.shared_residual) for d in data.dims]
        self.model = FactorCLSSL(
            encoders=encoders, feat_dims=[args.embed_dim, args.embed_dim],
            y_ohe_dim=1,
            activation='relu',
            lr=args.lr,
        ).to(args.dev)

        # separate Adams: non_club for encoders + heads, club for the CLUB upper-bound estimators
        self.non_club, self.club = self.model.get_optims()

    def loss(self, views):
        loss = self.model(*views)
        return loss, {'loss': loss.item()}

    def train_step(self, views):
        """One step on the main FactorCL loss, then n_club steps fitting the CLUB estimators."""
        loss, logs = self.loss(views)

        for o in self.non_club:
            o.zero_grad()
        loss.backward()
        for o in self.non_club:
            o.step()

        for _ in range(self.args.n_club):
            club_loss = self.model.learning_loss(*views)
            for o in self.club:
                o.zero_grad()
            club_loss.backward()
            for o in self.club:
                o.step()

        return logs

    @torch.no_grad()
    def embed(self, x1, x2):
        """
        s: heads of FactorCL's shared terms, I(X1;X2) lower bound + I(X1;X2|y) upper bound.
        p: heads of its unique terms, I(Xi;y) + I(X1;X2) upper bound + I(X1;X2|y) lower bound.
        """
        m = self.model
        m.eval()

        b1, b2 = m.backbones[0](x1), m.backbones[1](x2)
        s1 = torch.cat([m.linears_infonce_x1x2[0](b1), m.linears_club_x1x2_cond[0](b1)], 1)
        s2 = torch.cat([m.linears_infonce_x1x2[1](b2), m.linears_club_x1x2_cond[1](b2)], 1)
        p1 = torch.cat([m.linears_infonce_x1y(b1), m.linears_club_x1x2[0](b1), m.linears_infonce_x1x2_cond[0](b1)], 1)
        p2 = torch.cat([m.linears_infonce_x2y(b2), m.linears_club_x1x2[1](b2), m.linears_infonce_x1x2_cond[1](b2)], 1)

        return s1, p1, s2, p2

    @torch.no_grad()
    def embed_shared(self, x, modality):
        """Input of FactorCL's I(X1;X2) InfoNCE critic for one modality, used for retrieval."""
        m = self.model
        m.eval()
        b = m.backbones[modality](x)
        return m.linears_infonce_x1x2[modality](b)

    @torch.no_grad()
    def similarity(self, a, b):
        """Score of the I(X1;X2) concat critic f([a_i, b_j]) for every pair; f's first Linear is split over a and b."""
        f = self.model.infonce_x1x2._f
        d = a.shape[1]
        u = a @ f[0].weight[:, :d].t() + f[0].bias
        v = b @ f[0].weight[:, d:].t()
        return torch.cat([f[1:](u[i:i + 32, None] + v[None]).squeeze(-1) for i in range(0, len(u), 32)])
