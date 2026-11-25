from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, r2_score
from sklearn.preprocessing import StandardScaler

from datasets.clevr4 import TASKS
from datasets.rococo import DISTRACTORS, IMG_PARTS, MIX_SEEDS, ROCOCO_SETS, TXT_PARTS
from datasets.vggsound import CLASSES
from models.base import mlp
from utils import THIRD_PARTY, import_upstream


# --------------------------------------------------------------------------- shared


BLOCKS = ['s1', 'p1', 's2', 'p2']
BLOCK_PAIRS = [('s1', 's2'), ('p1', 'p2'), ('s1', 'p1'), ('s2', 'p2')]


def cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    return float(np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


# --------------------------------------------------------------------------- clevr4


def evaluate_clevr4(method, data):
    """
    acc: logreg accuracy of color / shape / texture from each block (s1, p1, s2, p2), 
        the concatenation s1_s2 = [s1, s2] and, for methods with an anchor, a = normalized mean of unit-norm s1, s2; 
        fit train, eval test.
    cka: linear CKA between block pairs; eval test.
    """
    Z = {}
    for split in ['train', 'test']:
        s1, p1, s2, p2 = method.embed(data.features(split, 'image'), data.features(split, 'text'))
        Z[split] = {name: t.cpu().numpy() for name, t in zip(BLOCKS + ['s1_s2'], (s1, p1, s2, p2, torch.cat([s1, s2], dim=-1)))}
        if method.has_anchor:
            Z[split]['a'] = F.normalize((F.normalize(s1, dim=-1) + F.normalize(s2, dim=-1)) / 2.0, dim=-1).cpu().numpy()

    acc = {}
    for name in Z['train']:
        scaler = StandardScaler().fit(Z['train'][name])
        z_train, z_test = scaler.transform(Z['train'][name]), scaler.transform(Z['test'][name])
        for task in TASKS:
            y_train, y_test = (data.labels[s][task].astype(str).values for s in ['train', 'test'])
            probe = LogisticRegression(max_iter=data.args.probe_max_iter).fit(z_train, y_train)
            acc[f'{name}_{task}'] = float(probe.score(z_test, y_test))

    return {'acc': acc, 'cka': {f'{a}-{b}': cka(Z['test'][a], Z['test'][b]) for a, b in BLOCK_PAIRS}}


# --------------------------------------------------------------------------- rococo


N_COCO_TEST = 5000


def evaluate_rococo(method, data):
    """
    retrieval on shared embeddings scored by method.similarity (cosine; FactorCL: its critic), eval test:
      coco_*: COCO 5k image->text and text->image R@1 / 5 / 10 / mean.
      rand_voca_* / same_concept_* / diff_concept_* / danger_*:
        image->text R@k and false_rate (top-1 is an adversarial caption).
      mix_0.9_* / mix_0.8_* / patch_0.9_* / patch_0.8_*:
        text->image R@k and false_rate (top-1 is a distractor image), mean over seeds.
    cka: linear CKA between block pairs; eval test.
    """
    meta = data.meta
    img = {p: method.embed_shared(data.features(p, 'image'), 0) for p in IMG_PARTS}
    txt = {p: method.embed_shared(data.features(p, 'text'), 1) for p in TXT_PARTS}

    coco = method.similarity(img['test'], txt['test']).cpu().numpy()
    retrieval = {f'coco_{k}': v for k, v in i2t_itm_eval(coco, meta['test']['img2txt'], []).items() if k != 'false_rate'}
    retrieval.update({f'coco_{k}': v for k, v in t2i_itm_eval(coco, meta['test']['txt2img'], None).items() if k != 'false_rate'})

    for name in ROCOCO_SETS:
        part = f'rococo_{name}'
        sims = method.similarity(img['test'], txt[part]).cpu().numpy()
        scores = i2t_itm_eval(sims, meta[part]['img2txt'], meta[part]['wrongtext'])
        retrieval.update({f'{name}_{k}': v for k, v in scores.items()})

    for d in DISTRACTORS:
        for s in MIX_SEEDS:
            gallery = torch.cat([img['test'], img[f'{d}_seed{s}']])
            scores = t2i_itm_eval(method.similarity(gallery, txt['test']).cpu().numpy(), meta['test']['txt2img'], None)
            retrieval.update({f'{d}_seed{s}_{k}': v for k, v in scores.items()})
        for k in ['img_r1', 'img_r5', 'img_r10', 'img_r_mean', 'false_rate']:
            retrieval[f'{d}_{k}'] = float(np.mean([retrieval[f'{d}_seed{s}_{k}'] for s in MIX_SEEDS]))

    for k, v in retrieval.items():
        print(f'{k:<22}{v:>10.2f}')

    e1, e2 = data.features('test', 'image')[meta['test']['txt2img']], data.features('test', 'text')
    blocks = dict(zip(BLOCKS, (t.cpu().numpy() for t in method.embed(e1, e2))))
    return {'retrieval': retrieval, 'cka': {f'{a}-{b}': cka(blocks[a], blocks[b]) for a, b in BLOCK_PAIRS}}


def i2t_itm_eval(sims, img2txt, wrongtext):
    """Image->text recall; RoCOCO original code."""
    ranks = np.zeros(sims.shape[0])
    false_cnt = 0

    for index, score in enumerate(sims):
        inds = np.argsort(score)[::-1]
        if inds[0] in wrongtext:
            false_cnt += 1
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    false_rate = 100 * false_cnt / len(ranks)
    tr_mean = (tr1 + tr5 + tr10) / 3

    eval_result = {'txt_r1': tr1, 'txt_r5': tr5, 'txt_r10': tr10, 'txt_r_mean': tr_mean, 'false_rate': false_rate}
    return eval_result


def t2i_itm_eval(sims, txt2img, img2txt):
    """Text->image recall; RoCOCO original code."""
    false_cnt = 0
    sims = sims.transpose()
    ranks = np.zeros(sims.shape[0])

    for index, score in enumerate(sims):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]
        if inds[0] >= N_COCO_TEST:
            false_cnt += 1

    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    false_rate = 100 * false_cnt / len(ranks)
    ir_mean = (ir1 + ir5 + ir10) / 3

    eval_result = {'img_r1': ir1, 'img_r5': ir5, 'img_r10': ir10, 'img_r_mean': ir_mean, 'false_rate': false_rate}
    return eval_result


# --------------------------------------------------------------------------- synthetic


def evaluate_synthetic(method, data):
    """
    acc: logreg accuracy of each latent's label (z1 / zs / z2 / ...) from each block (s1, p1, s2, p2, ...),
        the concatenation s1_s2 = [s1, s2, ...] and, for methods with an anchor, a = normalized mean of unit-norm s1, s2, ...;
        fit train, eval test.
    r2: linear-regression R2 of each latent from the same blocks;
        fit train, eval test.
    cka: linear CKA between block pairs (shared pairs, private pairs, s_m-p_m); eval test.
    """
    modalities = range(1, len(data.dims) + 1)
    blocks = [f'{k}{m}' for m in modalities for k in ('s', 'p')]
    block_pairs = ([(f's{i}', f's{j}') for i, j in combinations(modalities, 2)] + [(f'p{i}', f'p{j}') for i, j in combinations(modalities, 2)]
                   + [(f's{m}', f'p{m}') for m in modalities])

    Z = {}
    for split in ['train', 'test']:
        embeddings = method.embed(*data.x[split])
        ss = embeddings[0::2]
        Z[split] = {name: t.cpu().numpy() for name, t in zip(blocks + ['_'.join(blocks[0::2])], (*embeddings, torch.cat(ss, dim=-1)))}
        if method.has_anchor:
            Z[split]['a'] = F.normalize(sum(F.normalize(s, dim=-1) for s in ss) / len(ss), dim=-1).cpu().numpy()

    acc, r2 = {}, {}
    for name in Z['train']:
        scaler = StandardScaler().fit(Z['train'][name])
        z_train, z_test = scaler.transform(Z['train'][name]), scaler.transform(Z['test'][name])
        for latent in data.labels['train']:
            probe = LogisticRegression(max_iter=data.args.probe_max_iter).fit(z_train, data.labels['train'][latent])
            acc[f'{name}_{latent}'] = float(probe.score(z_test, data.labels['test'][latent]))
            linreg = LinearRegression().fit(z_train, data.latents['train'][latent])
            r2[f'{name}_{latent}'] = float(r2_score(data.latents['test'][latent], linreg.predict(z_test)))

    print(pd.DataFrame({'acc': acc, 'r2': r2}).round(3).to_string(), flush=True)
    return {'acc': acc, 'r2': r2, 'cka': {f'{a}-{b}': cka(Z['test'][a], Z['test'][b]) for a, b in block_pairs}}


# --------------------------------------------------------------------------- vggsound


VGGSOUNDER_MODALITIES = ['a', 'v', 'av', 'a only', 'v only']   # label subsets: a = A + AV labels, v = V + AV, ...
VGGSOUNDER_PURE_CLIPS = {'pure a': 'A', 'pure v': 'V', 'pure av': 'AV'}   # clips whose labels all carry this tag
VGGSOUNDER_METRICS = {'map': 'auc_pr',
                      'f1@1': 'f1@1',
                      'f1@3': 'f1@3',
                      'f1@5': 'f1@5',
                      'hit@1': 'hit_rate@1',
                      'hit@3': 'hit_rate@3',
                      'hit@5': 'hit_rate@5'}


def evaluate_vggsound(method, data):
    """
    Scores: logistic probe (cuML on the GPU, scaled like StandardScaler) on the VGGSound class from each block (s1, p1, s2, p2),
        the concatenations s1_s2, p1_p2, s1_p1, s2_p2, s1_s2_p1_p2 and, for methods with an anchor,
        a = normalized mean of unit-norm s1, s2 and its concatenations a_p1, a_p2, a_p1_p2; fit train, eval test.
    vggsound: top-1 accuracy and macro mAP on VGGSound test, one label per clip.
    vggsounder: mAP, micro F1@k and hit@k (k = 1, 3, 5) on the VGGSounder clips without background music (as the paper),
        against each label subset in VGGSOUNDER_MODALITIES and on the clips in VGGSOUNDER_PURE_CLIPS (all labels of the clip
        audio-only, video-only or audio-video); computed with VGGSounder's own benchmark code.
    reconstruction_gain: per modality m, R^2 of 2-layer MLP decoders (standardized in and out, fit train, eval test) reconstructing
        the pooled PE-AV features x_m from s_m, p_m, [s_m, p_m] and, for methods with an anchor, a and [a, p_m] (keys x{m}_{block}_r2);
        rg_m = R^2([s_m, p_m]) - max(R^2(s_m), R^2(p_m)), rg_a_m the same with a in place of s_m.
    cka: linear CKA between block pairs; eval test.
    """
    from cuml.linear_model import LogisticRegression as GPULogisticRegression   # imported here so other datasets never load cuML
    from cuml.preprocessing import StandardScaler as GPUStandardScaler

    benchmark = import_upstream(f'{THIRD_PARTY}/VGGSounder', 'vggsounder.benchmark')
    vggsounder_labels = benchmark.VGGSounder()
    ground_truth = {m: benchmark.get_gt_for_modality(vggsounder_labels, m) for m in VGGSOUNDER_MODALITIES}
    clip_tags = {v.video_id: (v.labels, {t.upper() for t in v.modalities}) for v in benchmark.VGGSounder()}
    ground_truth.update({name: pd.DataFrame([{'id': i, 'labels': labels} for i, (labels, tags) in clip_tags.items() if tags == {tag}])
                         for name, tag in VGGSOUNDER_PURE_CLIPS.items()})

    Z = {}
    for split in ['train', 'test']:
        s1, p1, s2, p2 = method.embed(*data.x[split])
        Z[split] = dict(zip(BLOCKS + ['s1_s2', 'p1_p2', 's1_p1', 's2_p2', 's1_s2_p1_p2'],
                            (s1, p1, s2, p2, torch.cat([s1, s2], dim=-1), torch.cat([p1, p2], dim=-1), torch.cat([s1, p1], dim=-1),
                             torch.cat([s2, p2], dim=-1), torch.cat([s1, s2, p1, p2], dim=-1))))
        if method.has_anchor:
            a = F.normalize((F.normalize(s1, dim=-1) + F.normalize(s2, dim=-1)) / 2.0, dim=-1)
            Z[split].update({'a': a, 'a_p1': torch.cat([a, p1], dim=-1), 'a_p2': torch.cat([a, p2], dim=-1),
                             'a_p1_p2': torch.cat([a, p1, p2], dim=-1)})

    vggsound, vggsounder, rows = {}, {}, []
    y_test = data.labels['test']
    for name in Z['train']:
        scaler = GPUStandardScaler().fit(Z['train'][name])
        z_train, z_test = scaler.transform(Z['train'][name]), scaler.transform(Z['test'][name])
        probe = GPULogisticRegression(max_iter=data.args.probe_max_iter, tol=1e-6, output_type='numpy')
        scores = probe.fit(z_train, data.labels['train'].astype(np.int32)).predict_proba(z_test)

        vggsound[f'{name}_acc'] = float((scores.argmax(1) == y_test).mean())
        vggsound[f'{name}_map'] = float(average_precision_score(np.eye(len(CLASSES))[y_test], scores, average='macro'))

        logits = pd.DataFrame({'id': data.ids['test'], 'logits': list(scores)})
        for m in ground_truth:
            results = benchmark.compare_labels_logits(ground_truth[m], logits, CLASSES, ks=[1, 3, 5])
            rows.append({'block': name, 'labels': m, **{metric: float(results[key]) for metric, key in VGGSOUNDER_METRICS.items()}})
            vggsounder.update({f'{name}_{m}_{metric}': rows[-1][metric] for metric in VGGSOUNDER_METRICS})

    reconstruction_gain = {}
    for m, (modality, s, p) in enumerate([('video', 's1', 'p1'), ('audio', 's2', 'p2')], start=1):
        x = {split: torch.load(f'{data.args.features_dir}/{split}/{modality}.pt').float().to(data.args.dev) for split in ['train', 'test']}
        x_mean, x_std = x['train'].mean(0), x['train'].std(0) + 1e-6
        y_train, y_test = (x['train'] - x_mean) / x_std, (x['test'] - x_mean) / x_std
        for name in [s, p, f'{s}_{p}'] + (['a', f'a_{p}'] if method.has_anchor else []):
            z_mean, z_std = Z['train'][name].mean(0), Z['train'][name].std(0) + 1e-6
            z_train, z_test = (Z['train'][name] - z_mean) / z_std, (Z['test'][name] - z_mean) / z_std
            decoder = mlp(z_train.shape[1], 512, y_train.shape[1], 2, 'relu').to(data.args.dev)
            optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
            for _ in range(100):
                for b in torch.randperm(len(z_train), device=z_train.device).split(1024):
                    loss = F.mse_loss(decoder(z_train[b]), y_train[b])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                r2 = 1 - ((decoder(z_test) - y_test) ** 2).sum(0) / ((y_test - y_test.mean(0)) ** 2).sum(0)
            reconstruction_gain[f'x{m}_{name}_r2'] = float(r2.mean())
        for shared, key in [(s, f'rg_{m}')] + ([('a', f'rg_a_{m}')] if method.has_anchor else []):
            reconstruction_gain[key] = reconstruction_gain[f'x{m}_{shared}_{p}_r2'] - max(reconstruction_gain[f'x{m}_{shared}_r2'], reconstruction_gain[f'x{m}_{p}_r2'])

    print(pd.Series(vggsound).round(4).to_string(), flush=True)
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
    print(pd.Series(reconstruction_gain).round(4).to_string(), flush=True)
    return {'vggsound': vggsound, 'vggsounder': vggsounder, 'reconstruction_gain': reconstruction_gain,
            'cka': {f'{a}-{b}': cka(Z['test'][a].cpu().numpy(), Z['test'][b].cpu().numpy()) for a, b in BLOCK_PAIRS}}


# --------------------------------------------------------------------------- registry


EVALUATE = {'clevr4': evaluate_clevr4,
            'vggsound': evaluate_vggsound,
            'rococo': evaluate_rococo,
            'synthetic': evaluate_synthetic}

SUMMARY_GROUPS = {
    'clevr4': {'final_losses': 'final_losses', 'acc': 'acc', 'cka': 'cka'},
    'vggsound': {g: g for g in ['final_losses', 'vggsound', 'vggsounder', 'reconstruction_gain', 'cka']},
    'rococo': {'final_losses': 'final_losses', 'retrieval': 'retrieval', 'cka': 'cka'},
    'synthetic': {g: g for g in ['final_losses', 'acc', 'r2', 'cka']},
}
