import torch

from utils import THIRD_PARTY, import_upstream
from .base import Method

repercent, perceiver = import_upstream(f'{THIRD_PARTY}/RePercENT', 'src.models.repercent', 'src.models.perceiver')

# Perceiver latent index of each component, per modality (upstream's 2-modality mapping)
DISEN_MAPPING = {'M_1': {'U_12': 0, 'S_12': 1}, 'M_2': {'U_21': 0, 'S_21': 1}}


class RePercENT(Method):
    """
    Upstream RePercENT, 2 modalities: per modality a Perceiver (group slot attention) whose two latents, read from the input
    tokens, are the private u and shared s; upstream DisenLoss on the sample and its augmented view, with a fixed lmd
    and Adam weight decay 1e-4 (as upstream's IRFL config). Inputs: token features [n, tokens, dim] (args.tokens),
    all-zero tokens are padding and masked out; or pooled features [n, dim], as one token.
    """
    uses_views = True   # batches are (x1, x2, v1, v2), v = augmented views

    def __init__(self, args, data):
        super().__init__(args, data)
        encoders = [repercent.DisenEncoder(torch.nn.Identity(), perceiver.Perceiver(
            depth=args.depth, input_channels=dim, input_axis=1, num_latents=2, latent_dim=args.embed_dim,
            cross_heads=args.cross_heads, cross_dim_head=args.embed_dim // args.cross_heads,
            latent_heads=args.latent_heads, latent_dim_head=args.embed_dim // args.latent_heads,
            fourier_encode_data=False, use_slot_attn=True)) for dim in data.dims]
        self.model = repercent.RePercENT(M=2, disenEncoder=encoders, disen_mapping=DISEN_MAPPING, vmfkappa=args.kappa).to(args.dev)
        self.disen_loss = repercent.DisenLoss(alpha=args.alpha, beta=args.beta, lmd=args.lmd, lmd_end_value=0, M=2).to(args.dev)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-4)

    def tokens(self, x):
        """Float tokens on args.dev ([n, dim] -> [n, 1, dim]) and the mask of non-padding tokens."""
        x = x.to(self.args.dev).float()
        x = x[:, None] if x.dim() == 2 else x
        return x, (x != 0).any(-1)

    def loss(self, views):
        (x1, m1), (x2, m2), (v1, n1), (v2, n2) = (self.tokens(x) for x in views)
        outputs = self.model([x1, x2], mask=[m1, m2])
        outputs_aug = self.model([v1, v2], mask=[n1, n2])
        return self.disen_loss(outputs, outputs_aug)

    @torch.no_grad()
    def embed(self, x1, x2):
        """(s1, p1, s2, p2) = upstream's S_view and U latents, in batches of args.batch_size."""
        self.model.eval()
        blocks = []
        for i in range(0, len(x1), self.args.batch_size):
            (a, mask_a), (b, mask_b) = self.tokens(x1[i:i + self.args.batch_size]), self.tokens(x2[i:i + self.args.batch_size])
            out = self.model([a, b], mask=[mask_a, mask_b])
            blocks.append((out['S_view'][:, 0, 1], out['U'][:, 0, 1], out['S_view'][:, 1, 0], out['U'][:, 1, 0]))
        return tuple(torch.cat(block) for block in zip(*blocks))

    @torch.no_grad()
    def embed_shared(self, x, modality):
        """S_view latent of one modality alone (its Perceiver, with the same latent position encodings as forward), used for retrieval."""
        self.model.eval()
        m = modality + 1
        pos_enc = (self.model.pair_pos_enc[getattr(self.model, f'pair_idx_m{m}')]
                   + self.model.type_pos_enc[getattr(self.model, f'type_idx_m{m}')])
        slot = DISEN_MAPPING[f'M_{m}'][f'S_{m}{3 - m}']

        out = []
        for i in range(0, len(x), self.args.batch_size):
            tokens, mask = self.tokens(x[i:i + self.args.batch_size])
            out.append(self.model.disenEncoders[modality](tokens, mask=mask, pos_enc=pos_enc)[:, slot])
        return torch.cat(out)
