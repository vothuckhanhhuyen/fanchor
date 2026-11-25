import importlib

DATASETS = {   # dataset name on the command line -> module.Class in datasets/
    'clevr4': 'clevr4.CLEVR4',
    'vggsound': 'vggsound.VGGSound',
    'rococo': 'rococo.RoCOCO',
    'synthetic': 'synthetic.Synthetic',
}


def get_dataset(name):
    """Import only the requested dataset's module and return its class."""
    module, cls = DATASETS[name].split('.')
    return getattr(importlib.import_module(f'datasets.{module}'), cls)
