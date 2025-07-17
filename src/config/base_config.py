import os.path as osp
from typing import Union
from yacs.config import CfgNode as CN


_C = CN()

_C.PATH = CN()
_C.PATH.ROOT = '/Users/max/python_code/functional_connectivity'
_C.PATH.DATA = '/Volumes/PortableSSD/max/magistr/research/fMRI/sorted'


_C.BIDS = CN()
_C.BIDS.ADULTS_BIDS_ROOT = 'adults'  # Join with _C.PATH.DATA
_C.BIDS.TEENAGERS_BIDS_ROOT = 'teenagers'  # Join with _C.PATH.DATA
_C.BIDS.CHILDREN_BIDS_ROOT = 'yong_children'  # Join with _C.PATH.DATA
_C.BIDS.VALIDATE = False
_C.BIDS.DATATYPE = 'func'
_C.BIDS.TASK = 'rest'
_C.BIDS.SPACE = 'MNI152NLin2009cAsym'
_C.BIDS.RETURN_TYPE = 'file'
_C.BIDS.FUNC_SUFFIX = 'bold'
_C.BIDS.MASK_SUFFIX = 'mask'
_C.BIDS.CONFOUNDS_SUFFIX = 'timeseries'
_C.BIDS.FUNC_FILE_EXTENSION = '.nii.gz'
_C.BIDS.MASK_FILE_EXTENSION = '.nii.gz'
_C.BIDS.CONFOUNDS_FILE_EXTENSION = '.tsv'


_C.PARCELLATION = CN()
_C.PARCELLATION.ATLAS_FILE = 'resources/rois/schaefer_2018/schaefer_2018/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz'  # Join with _C.PATH.ROOT
_C.PARCELLATION.ATLAS_LABELS_FILE = 'resources/rois/schaefer_2018/schaefer_2018/Schaefer2018_100Parcels_7Networks_order.txt' # Join with _C.PATH.ROOT
_C.PARCELLATION.CONFOUNDS = ['trans_x', 'trans_y', 'trans_z',
                             'rot_x', 'rot_y', 'rot_z',
                              'white_matter', 'csf', 'global_signal']
_C.PARCELLATION.HIGH_PASS = 0.009
_C.PARCELLATION.LOW_PASS = 0.08
_C.PARCELLATION.DETREND = True
_C.PARCELLATION.STANDARDIZE = 'zscore_sample'
_C.PARCELLATION.TR_DROP = 1
_C.PARCELLATION.TR = 2
_C.PARCELLATION.MEMORY = 'nilearn_cache'
_C.PARCELLATION.RESULTS_OUT_FOLDER = 'results/parcellation/Schaefer2018_100Parcels_7Networks'  # Join with _C.PATH.ROOT
_C.PARCELLATION.RESULTS_FILE_NAME = 'parcellation.h5'

possible_measures = [
    'pearson',
    'cross_correlation_statmodels',
    'cross_correlation_numpy',
    'coherence',
    'wavelet_coherence',
    'mutual_information',
    'euclidean_distance',
    'cityblock_distance',
    'dynamic_time_warping',
    'earth_movers_distance',
    'percentage_bend_correlation',
    'spearman_rank_correlation',
    'partial_correlation']


_C.CONNECTIVITY = CN()
_C.CONNECTIVITY.MEASURES = [  # Should be in possible_measures
    'pearson',
    'cross_correlation_statmodels',
    #'cross_correlation_numpy',
    'coherence',
    'mutual_information',
    'euclidean_distance',
    'cityblock_distance',
    'earth_movers_distance',
    'wavelet_coherence',
    'dynamic_time_warping',
    'percentage_bend_correlation',
    'spearman_rank_correlation',
    'partial_correlation'
]
_C.CONNECTIVITY.RESULTS_OUT_FOLDER = 'results/connectivity/Schaefer2018_100Parcels_7Networks'
_C.CONNECTIVITY.RESULTS_FILE_NAME = 'connectivity.h5'


_C.GRAPH_FEATURES = CN()
_C.GRAPH_FEATURES.RESULTS_OUT_FOLDER = 'results/graph_features/Schaefer2018_100Parcels_7Networks'
_C.GRAPH_FEATURES.RESULTS_FILE_NAME = 'graph_features.h5'


_C.ML = CN()
_C.ML.ROOT = 'results/ml'  # Join with _C.PATH.ROOT
_C.ML.RAW_CONNECTIVITY_RESULTS_OUT_FOLDER = 'raw_connectivity'  # Join with _C.ML.ROOT
_C.ML.GRAPH_CONNECTIVITY_RESULTS_OUT_FOLDER = 'graph_connectivity'  # Join with _C.ML.ROOT
_C.ML.COMBINED_CONNECTIVITY_RESULTS_OUT_FOLDER = 'combined_connectivity'  # Join with _C.ML.ROOT

_C.RAW_CONNECTIVITY_SVC = CN()
_C.RAW_CONNECTIVITY_SVC.KERNEL = 'linear'
_C.RAW_CONNECTIVITY_SVC.C = 1.0
_C.RAW_CONNECTIVITY_SVC.RANDOM_STATE = 42

_C.RAW_CONNECTIVITY_FEATURE_EXTRACTOR = CN()
_C.RAW_CONNECTIVITY_FEATURE_EXTRACTOR.K_FEATURES = 50
_C.RAW_CONNECTIVITY_FEATURE_EXTRACTOR.CV_FOLDS = 5
_C.RAW_CONNECTIVITY_FEATURE_EXTRACTOR.METRIC = 'accuracy'

_C.GRAPH_CONNECTIVITY_SVC = CN()
_C.GRAPH_CONNECTIVITY_SVC.KERNEL = 'linear'
_C.GRAPH_CONNECTIVITY_SVC.C = 1.0
_C.GRAPH_CONNECTIVITY_SVC.RANDOM_STATE = 42

_C.GRAPH_CONNECTIVITY_FEATURE_EXTRACTOR = CN()
_C.GRAPH_CONNECTIVITY_FEATURE_EXTRACTOR.K_FEATURES = 30
_C.GRAPH_CONNECTIVITY_FEATURE_EXTRACTOR.CV_FOLDS = 5
_C.GRAPH_CONNECTIVITY_FEATURE_EXTRACTOR.METRIC = 'accuracy'

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

    base_config.PARCELLATION.ATLAS_LABELS_FILE = osp.join(
        base_config.PATH.ROOT,
        base_config.PARCELLATION.ATLAS_LABELS_FILE
    )

    base_config.PARCELLATION.RESULTS_OUT_FOLDER = osp.join(
        base_config.PATH.ROOT,
        base_config.PARCELLATION.RESULTS_OUT_FOLDER
    )

    base_config.CONNECTIVITY.RESULTS_OUT_FOLDER = osp.join(
        base_config.PATH.ROOT,
        base_config.CONNECTIVITY.RESULTS_OUT_FOLDER
    )

    base_config.GRAPH_FEATURES.RESULTS_OUT_FOLDER = osp.join(
        base_config.PATH.ROOT,
        base_config.GRAPH_FEATURES.RESULTS_OUT_FOLDER
    )

    base_config.ML.ROOT = osp.join(
        base_config.PATH.ROOT,
        base_config.ML.ROOT
    )

    base_config.ML.RAW_CONNECTIVITY_RESULTS_OUT_FOLDER = osp.join(
        base_config.ML.ROOT,
        base_config.ML.RAW_CONNECTIVITY_RESULTS_OUT_FOLDER
    )

    base_config.ML.GRAPH_CONNECTIVITY_RESULTS_OUT_FOLDER = osp.join(
        base_config.ML.ROOT,
        base_config.ML.GRAPH_CONNECTIVITY_RESULTS_OUT_FOLDER
    )

    base_config.ML.COMBINED_CONNECTIVITY_RESULTS_OUT_FOLDER = osp.join(
        base_config.ML.ROOT,
        base_config.ML.COMBINED_CONNECTIVITY_RESULTS_OUT_FOLDER
    )

    return base_config
