import importlib

METHODS = {
    'fanchor': 'fanchor.FAnchor',
    'factorcl': 'factorcl.FactorCL',
    'disentangledssl': 'disentangledssl.DisentangledSSL',
    'jointdisen': 'jointdisen.JointDisen',
    'repercent': 'repercent.RePercENT',
    'finetuned_proj_clip': 'finetuned_proj_clip.FinetunedProjCLIP',
}


def get_method(name):
    module, cls = METHODS[name].split('.')
    return getattr(importlib.import_module(f'models.{module}'), cls)
