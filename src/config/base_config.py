import os.path as osp
from typing import Union
from yacs.config import CfgNode as CN


_C = CN()

_C.PATH = CN()
_C.PATH.ROOT = '/Users/max/python_code/functional_connectivity'
_C.PATH.DATA = '/Volumes/PortableSSD/max/magistr/research/fMRI/sorted'


_C.BIDS = CN()
_C.BIDS.ADULTS_BIDS_ROOT = 'adults/'
_C.BIDS.TEENAGERS_BIDS_ROOT = 'teenagers/'
_C.BIDS.CHILDREN_BIDS_ROOT = 'yong_children/'
_C.BIDS.DATATYPE = 'func'
_C.BIDS.TASK = 'rest'
_C.BIDS.SPACE = 'MNI152NLin2009cAsym'
_C.BIDS.RETURN_TYPE = 'file'
_C.BIDS.FUNC_SUFFIX = 'bold'
_C.BIDS.MASK_SUFFIX = 'mask'
_C.BIDS.CONFOUNDS_SUFFIX = 'timeseries'


_C.PARCELLATION = CN()
_C.PARCELLATION.ATLAS_FILE = 'resources/rois/aal/aal_SPM12/aal/atlas/AAL.nii'
_C.PARCELLATION.CONFOUNDS = ['trans_x', 'trans_y', 'trans_z',
                             'rot_x', 'rot_y', 'rot_z',
                              'white_matter', 'csf', 'global_signal']
_C.PARCELLATION.HIGH_PASS = 0.009
_C.PARCELLATION.LOW_PASS = 0.08
_C.PARCELLATION.DETREND = True
_C.PARCELLATION.STANDARDIZE = True
_C.PARCELLATION.TR_DROP = 1
_C.PARCELLATION.TR = 2
_C.PARCELLATION.MEMORY = 'nilearn_cache'


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

    base_config.BIDS.ADULTS_BIDS_ROOT = osp.join(
        base_config.PATH.DATA,
        base_config.BIDS.ADULTS_BIDS_ROOT
    )

    base_config.BIDS.TEENAGERS_BIDS_ROOT = osp.join(
        base_config.PATH.DATA,
        base_config.BIDS.TEENAGERS_BIDS_ROOT
    )

    base_config.BIDS.CHILDREN_BIDS_ROOT = osp.join(
        base_config.PATH.DATA,
        base_config.BIDS.CHILDREN_BIDS_ROOT
    )

    base_config.PARCELLATION.ATLAS_FILE = osp.join(
        base_config.PATH.ROOT,
        base_config.PARCELLATION.ATLAS_FILE
    )

    return base_config
