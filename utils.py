import importlib
import json
import os
import random
import sys
import time
import uuid
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = f'{REPO}/results'
THIRD_PARTY = f'{REPO}/third_party'


def set_seed(seed):
    """Seed python, numpy and torch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_defaults(dataset, method):
    """Dataset config from configs/<dataset>.yaml with the method's section merged on top."""
    cfg = OmegaConf.load(f'{REPO}/configs/{dataset}.yaml')
    return OmegaConf.to_container(OmegaConf.merge(cfg, cfg.pop('methods')[method]))


def parse(argv):
    dataset, method = argv[0], argv[1]
    defaults = {
        **load_defaults(dataset, method),
        'results_path': f'{RESULTS_DIR}/{dataset}/{method}_run_results',
        'train_seed': 0,
        'dev': 'cuda:0',
        'summary_seeds': [0, 1, 2, 3, 4],
        'summary_path': None,
    }

    overrides = OmegaConf.to_container(OmegaConf.from_dotlist(argv[2:]))
    return dataset, method, defaults, SimpleNamespace(**{**defaults, **overrides})


# keys ignored when matching a run against saved results (paths, device, seeds, logging)
MATCH_EXCLUDE = {'train_seed',
                 'summary_seeds',
                 'results_path',
                 'summary_path',
                 'dev',
                 'data_root',
                 'features_dir',
                 'record_every'}


def append_result(results_dir, record):
    """Write one record as its own json file; tmp file + rename so parallel runs never see partial files."""
    os.makedirs(results_dir, exist_ok=True)
    name = f'{time.time_ns()}_{uuid.uuid4().hex[:8]}.json'
    with open(f'{results_dir}/.tmp_{name}', 'w') as f:
        json.dump(record, f)
    os.rename(f'{results_dir}/.tmp_{name}', f'{results_dir}/{name}')


def matching_results(results_dir, args, defaults):
    """Saved records whose config (filled with defaults) equals args, ignoring MATCH_EXCLUDE keys."""
    target = {k: v for k, v in vars(args).items() if k not in MATCH_EXCLUDE}

    if not os.path.isdir(results_dir):
        return []
    records = [json.load(open(f'{results_dir}/{f}')) for f in sorted(os.listdir(results_dir)) if not f.startswith('.')]

    return [r for r in records
            if {k: v for k, v in {**defaults, **r['config']}.items() if k not in MATCH_EXCLUDE} == target]


def import_upstream(root, *modules):
    """Import modules from a baseline repo whose module names clash with ours, then restore our sys.modules."""
    local = {os.path.splitext(n)[0] for n in os.listdir(root)}

    def clashes(name):
        return name.split('.')[0] in local

    ours = {n: sys.modules.pop(n) for n in list(sys.modules) if clashes(n)}
    sys.path.insert(0, root)
    imported = [importlib.import_module(m) for m in modules]
    sys.path.remove(root)
    for n in [n for n in sys.modules if clashes(n)]:
        del sys.modules[n]
    sys.modules.update(ours)

    return imported[0] if len(imported) == 1 else imported
