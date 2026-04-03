import importlib
from os import path as osp
from utils.utils import scandir

# automatically scan and import model modules
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
# model_filenames = [
#     osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
#     if v.endswith('_model.py')
# ]
# import all the model modules
# _model_modules = [
#     importlib.import_module(f'models.{file_name}')
#     for file_name in model_filenames
# ]


def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_name = opt['model_name']  # 3unet_model.py
    net_name = opt['net_name']  # SCASDnet

    model_py_file = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v == model_name + '.py']
    nets = [
        importlib.import_module(f'models.{file_name}')
        for file_name in model_py_file
    ]
    # dynamic instantiation
    for name in nets:
        net_cls = getattr(name, net_name, None)
        if net_cls is not None:
            break
    if net_cls is None:
        raise ValueError(f'Model {net_name} is not found.')

    model = net_cls(**opt['network'])

    return model


