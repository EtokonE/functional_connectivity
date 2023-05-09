import os.path as osp
from typing import Union
from yacs.config import CfgNode as CN


_C = CN()

_C.PATH.DATA = '/Volumes/PortableSSD/max/magistr/research/fMRI/'



def get_cfg_defaults():
    """Returns yacs CfgNode object"""
    return _C.clone()


def combine_config(cfg_path: Union[str, None] = None):
    """
    Merge base config file with the config file
    of certain experiment
    Args:
         cfg_path (str): file in .yaml or .yml format with
         config parameters or None to use Base config
    Returns:
        yacs CfgNode object
    """
    base_config = get_cfg_defaults()
    if cfg_path is not None:
        if osp.exists(cfg_path):
            base_config.merge_from_file(cfg_path)
        else:
            raise FileNotFoundError(f'File {cfg_path} does not exists')

    return base_config
