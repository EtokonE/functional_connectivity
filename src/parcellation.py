"""
This script is used to extract the time series from the preprocessed functional data
using the parcellation atlas. The atlas was chosen from:
- https://github.com/nilearn/nilearn/blob/main/nilearn/datasets/atlas.py
"""
import bids
from nilearn import maskers
from nilearn import image as nimg
from src.logger import get_logger
from yacs.config import CfgNode
from src.utils import extract_extension_files, extract_confounds
from src.config.base_config import combine_config


cfg = combine_config()
logger = get_logger(__name__)


def load_atlas(parcel_file: str) -> nimg.Nifti1Image:
    """Load the parcellation atlas"""
    parcel_atlas = nimg.load_img(parcel_file)
    return parcel_atlas


def define_masker(labels_img: str,
                  standardize: bool,
                  memory: str,
                  detrend: bool,
                  high_pass: float,
                  low_pass: float,
                  t_r: int,
                  **kwargs) -> maskers.NiftiLabelsMasker:
    """
    Define the masker object

    Args:
        labels_img (str): parcellation atlas
        standardize (bool): standardize the time series
        memory (str): caching directory
        detrend (bool): detrend the time series
        high_pass (float): high pass filter
        low_pass (float): low pass filter
        t_r (int): repetition time
    """
    masker = maskers.NiftiLabelsMasker(labels_img=labels_img,
                                       standardize=standardize,
                                       memory=memory,
                                       detrend=detrend,
                                       high_pass=high_pass,
                                       low_pass=low_pass,
                                       t_r=t_r,
                                       **kwargs)
    return masker


def get_layout(bids_root: str, validate=False, **kwargs) -> bids.BIDSLayout:
    """
    Get the BIDS layout object

    Args:
        bids_root (str): path to BIDS directory
        validate (bool): validate the BIDS dataset
    """
    layout = bids.BIDSLayout(bids_root, validate=validate, **kwargs)
    logger.info(
        f'Layout contains: {len(layout.get_subjects())} \n\
        List of subjects: {layout.get_subjects()} \n\n\
        List of tasks: {layout.get_tasks()}'
    )
    return layout







layout = bids.BIDSLayout(cfg.BIDS.ADULTS_BIDS_ROOT, validate=False)
print(f'List of subjects: {layout.get_subjects()} \n\nList of tasks: {layout.get_tasks()}')
print(layout)

# Get preprocessed functional files
func_files = layout.get(return_type=cfg.BIDS.RETURN_TYPE,
                        subject='001',
                        datatype=cfg.BIDS.DATATYPE,
                        suffix=cfg.BIDS.FUNC_SUFFIX,
                        space=cfg.BIDS.SPACE)

func_files = extract_extension_files(func_files)

print(func_files)


# Get resting state data (preprocessed, mask, and confounds file)
func_files = layout.get(return_type=cfg.BIDS.RETURN_TYPE,
                        task=cfg.BIDS.TASK,
                        datatype=cfg.BIDS.DATATYPE,
                        suffix=cfg.BIDS.FUNC_SUFFIX,
                        space=cfg.BIDS.SPACE)
func_files = extract_extension_files(func_files)

mask_files = layout.get(return_type=cfg.BIDS.RETURN_TYPE,
                        datatype=cfg.BIDS.DATATYPE,
                        task=cfg.BIDS.TASK,
                        suffix=cfg.BIDS.MASK_SUFFIX,
                        space=cfg.BIDS.SPACE)
mask_files = extract_extension_files(mask_files)

confound_files = layout.get(return_type=cfg.BIDS.RETURN_TYPE,
                            datatype=cfg.BIDS.DATATYPE,
                            task=cfg.BIDS.TASK,
                            suffix=cfg.BIDS.CONFOUNDS_SUFFIX)
confound_files = extract_extension_files(confound_files, extension='tsv')

print(func_files, '\n', len(func_files))
print(mask_files, '\n', len(mask_files))
print(confound_files, '\n', len(confound_files))

params = {
    'parcel_file': '../resources/rois/yeo_2011/Yeo_JNeurophysiol11_MNI152/relabeled_yeo_atlas.nii.gz',
    'confounds': ['trans_x', 'trans_y', 'trans_z',
                  'rot_x', 'rot_y', 'rot_z',
                  'white_matter', 'csf', 'global_signal'],
    'high_pass': 0.009,
    'low_pass': 0.08,
    'detrend': True,
    'standardize': True,
    'tr_drop': 4

}

yeo_7 = nimg.load_img(params['parcel_file'])

masker = maskers.NiftiLabelsMasker(labels_img=yeo_7,
                                   standardize=params['standardize'],
                                   memory='nilearn_cache',
                                   detrend=params['detrend'],
                                   low_pass=params['low_pass'],
                                   high_pass=params['high_pass'],
                                   t_r=2)


# Pull the first subject's data
func_file = func_files[0]
mask_file = mask_files[0]
confound_file = confound_files[0]


print(func_file, mask_file, confound_file)

# Load func image
func_img = nimg.load_img(func_file)

# Remove the first 4 TRs
func_img = func_img.slicer[:, :, :, params['tr_drop']:]

# Use the above function to pull out a confound matrix
confounds = extract_confounds(confound_file,
                              params['confounds'])
# Drop the first 4 rows of the confounds matrix
confounds = confounds[params['tr_drop']:, :]


# Apply cleaning, parcellation and extraction to functional data
cleaned_and_averaged_time_series = masker.fit_transform(func_img, confounds)
print(cleaned_and_averaged_time_series.shape)

print(type(cleaned_and_averaged_time_series))