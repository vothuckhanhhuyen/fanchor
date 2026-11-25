import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Method:
    """Base for all methods: subclasses set self.model / self.optimizer and implement loss, embed, embed_shared."""
    uses_views = False   # True if training batches need augmented views
    has_anchor = False   # True if the method trains a shared anchor (evaluated as block a)
    scheduler = None     # stepped once per epoch if set

    def __init__(self, args, data):
        self.args = args

    def loss(self, batch):
        """Return (loss tensor to backprop, dict of floats to log); used by both train_step and val_step."""
        raise NotImplementedError

    def train_step(self, batch):
        loss, logs = self.loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return logs

    def val_step(self, batch):
        return self.loss(batch)[1]

    def fit(self, data):
        hist = train_loop(self.model, data, self.args.epochs, self.train_step, self.val_step,
                          views=self.uses_views, record_every=self.args.record_every,
                          end_epoch=self.scheduler.step if self.scheduler else None)

        return {'final_losses': {k: v for k, v in hist[-1].items() if k != 'epoch'}, 'hist': hist}

    def embed(self, *xs):
        """One input per modality -> the blocks evaluate uses (e.g. s1, p1, s2, p2)."""
        raise NotImplementedError

    def embed_shared(self, x, modality):
        """Shared embedding of one modality (by index), used for retrieval."""
        raise NotImplementedError

    def similarity(self, a, b):
        """Retrieval scores between every row of a and every row of b (embed_shared outputs): cosine."""
        return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).t()


def train_loop(model, data, epochs, train_step, val_step, views=False, record_every=1, end_epoch=None, name=''):
    """Train every epoch; every record_every epochs (and the last) log mean train / val logs (val weighted by batch size)."""
    hist = []
    for epoch in range(epochs):
        model.train()
        train_logs = [train_step(batch) for batch in data.batches('train', True, views)]
        if end_epoch:
            end_epoch()

        if not (epoch % record_every == 0 or epoch == epochs - 1):
            continue
        row = {'epoch': epoch, **{f'train_{k}': float(np.mean([log[k] for log in train_logs])) for k in train_logs[0]}}

        model.eval()
        with torch.no_grad():
            val_logs, sizes = [], []
            for batch in data.batches('val', False, views):
                val_logs.append(val_step(batch))
                sizes.append(len(batch[0]))
        row.update({f'val_{k}': float(np.average([log[k] for log in val_logs], weights=sizes)) for k in val_logs[0]})

        hist.append(row)
        print(name, row, flush=True)

    return hist


def mlp(dim, hidden_dim, output_dim, layers, activation):
    """`layers` Linear layers in total (as upstream DisentangledSSL): dim -> hidden, layers - 2 x hidden -> hidden, hidden -> output."""
    activation = {
        'relu': nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
        'tanh': nn.Tanh,
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers - 2):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


def infonce_loss(z1, z2, tau=0.07):
    """Symmetric InfoNCE on normalized z1, z2, split into (alignment, uniformity) terms."""
    pos = (z1 * z2).sum(-1)
    sim_12 = z1 @ z2.t()
    lse_12 = torch.logsumexp(sim_12 / tau, dim=1)
    lse_21 = torch.logsumexp(sim_12.t() / tau, dim=1)

    loss_align = (-pos / tau).mean()
    loss_uniform = 0.5 * (lse_12.mean() + lse_21.mean())
    return loss_align, loss_uniform


def multimodal_contrastive_loss(zs, tau, loss_type):
    """(alignment, uniformity) over M modalities: fanchor, centrobind, or mean pairwise InfoNCE."""
    # each z pulled to its sample's centroid; centroids pushed apart
    if loss_type == 'fanchor':
        z = torch.stack(zs, dim=1)
        centroid = F.normalize(z.mean(1), dim=-1)
        pos = (z * centroid.unsqueeze(1)).sum(-1)
        lse = torch.logsumexp(centroid @ centroid.t() / tau, dim=1)
        return (-pos / tau).mean(), lse.mean()

    # InfoNCE between the centroid and each modality
    if loss_type == 'centrobind':
        centroid = F.normalize(torch.stack(zs, dim=1).mean(1), dim=-1)
        terms = [infonce_loss(centroid, z, tau=tau) for z in zs]
        return sum(a for a, _ in terms), sum(u for _, u in terms)

    # default: InfoNCE averaged over all modality pairs
    pairs = [infonce_loss(zi, zj, tau=tau) for i, zi in enumerate(zs) for zj in zs[i + 1:]]
    return sum(a for a, _ in pairs) / len(pairs), sum(u for _, u in pairs) / len(pairs)
