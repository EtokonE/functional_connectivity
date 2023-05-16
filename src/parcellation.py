"""
This script is used to extract the time series from the preprocessed functional data
using the parcellation atlas. The atlas was chosen from:
- https://github.com/nilearn/nilearn/blob/main/nilearn/datasets/atlas.py
"""
import os
import bids
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from yacs.config import CfgNode
from nilearn import maskers
from nilearn import image as nimg
from typing import List, Tuple
from src.logger import get_logger
from src.utils import extract_extension_files, load_fimg, extract_confound_vars, get_time_series
from src.config.base_config import combine_config


logger = get_logger('parcellation')


def load_atlas(parcel_file: str):
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

    Returns:
        masker (NiftiLabelsMasker): masker object
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

    Returns:
        layout (BIDSLayout): BIDS layout object
    """
    layout = bids.BIDSLayout(bids_root, validate=validate, **kwargs)
    logger.info(
        f'Layout contains: {len(layout.get_subjects())} \n\
        List of subjects: {layout.get_subjects()} \n\n\
        List of tasks: {layout.get_tasks()}'
    )
    return layout


def get_files(layout: bids.BIDSLayout,
              subject: str,
              return_type: str,
              task: str,
              datatype: str,
              suffix: str,
              file_extension: str,
              **kwargs) -> list:
    """
    Get the preprocessed files from layout

    Args:
         layout (BIDSLayout): BIDS layout object
         subject (str): subject id
         return_type (str): return type (object/file/id)
         task (str): task (rest)
         datatype (str): data type
         suffix (str): suffix
         file_extension (str): file extension

    Returns:
         files (list): list of files
    """
    files = layout.get(subject=subject,
                       return_type=return_type,
                       task=task,
                       datatype=datatype,
                       suffix=suffix,
                       **kwargs)
    file = extract_extension_files(files, file_extension)[0]
    logger.info(f'Process {file} file')
    return file


def extract_confounds(confound_tsv: str, confounds: list, dt: bool = True) -> np.ndarray:
    """
    Extract confound matrix from tsv file

    Arguments:
        confound_tsv                   Full path to confounds.tsv
        confounds                      A list of confounder variables to extract
        dt                             Compute temporal derivatives [default = True]

    returns:
        confound_matrix                 
    """

    if dt:
        dt_names = ['{}_derivative1'.format(c) for c in confounds]
        confounds = confounds + dt_names

    # Extract relevant columns
    confound_df = pd.read_csv(confound_tsv, delimiter='\t')
    confound_df = confound_df[confounds]

    # Convert into a matrix of values (timepoints)x(variable)
    confound_mat = confound_df.values

    return confound_mat


def define_h5_path(out_folder: str, file_name: str, age_group: str):
    """Define the full path for the results file"""
    full_path = os.path.join(out_folder, f'{age_group}_{file_name}')
    return full_path


def save_parcellation_results(time_series: List[np.ndarray],
                              labels_list: List[np.ndarray],
                              out_folder: str = 'results',
                              file_name: str = 'parcellation.h5',
                              age_group: str = 'adults') -> None:
    """
    Save the parcellation results in h5 file

    Args:
        time_series (List[np.ndarray]): time series
        labels_list (List[np.ndarray]): list of atlas ROIs
        out_folder (str): output folder where to save the results file
        file_name (str): name of the results file
        age_group (str): age group (adults/children...) to add to the file name
    """
    full_path = define_h5_path(out_folder, file_name, age_group)
    os.makedirs(out_folder, exist_ok=True)

    with h5py.File(full_path, 'w') as f:
        f.create_dataset('time_series', data=np.array(time_series))
        f.create_dataset('labels_list', data=np.array(labels_list))

    logger.info(f'Parcellation results for age group {age_group} saved in {full_path}')


def extract_time_series(layout: bids.BIDSLayout,
                        masker: maskers.NiftiLabelsMasker,
                        config: CfgNode) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract the time series from the preprocessed functional data, using the parcellation atlas

    Args:
        layout (BIDSLayout): BIDS layout object
        masker (NiftiLabelsMasker): masker object
        config (CfgNode): configuration file

    Returns:
        time_series (List[np.ndarray]): time series
        labels_list (List[np.ndarray]): list of labels
    """
    subjects_list = layout.get_subjects()
    logger.info(f'Found {len(subjects_list)} subjects\n {subjects_list}')

    time_series = []
    labels_list = []

    for subject in tqdm(subjects_list):
        logger.info(f'Extracting time series for subject: {subject}')
        func_file = get_files(layout=layout,
                              subject=subject,
                              return_type=config.BIDS.RETURN_TYPE,
                              task=config.BIDS.TASK,
                              datatype=config.BIDS.DATATYPE,
                              suffix=config.BIDS.FUNC_SUFFIX,
                              file_extension=config.BIDS.FUNC_FILE_EXTENSION,
                              space=config.BIDS.SPACE)

        mask_file = get_files(layout=layout,
                              subject=subject,
                              return_type=config.BIDS.RETURN_TYPE,
                              task=config.BIDS.TASK,
                              datatype=config.BIDS.DATATYPE,
                              suffix=config.BIDS.MASK_SUFFIX,
                              file_extension=config.BIDS.MASK_FILE_EXTENSION,
                              space=config.BIDS.SPACE)

        confound_file = get_files(layout=layout,
                                  subject=subject,
                                  return_type=config.BIDS.RETURN_TYPE,
                                  task=config.BIDS.TASK,
                                  datatype=config.BIDS.DATATYPE,
                                  suffix=config.BIDS.CONFOUNDS_SUFFIX,
                                  file_extension=config.BIDS.CONFOUNDS_FILE_EXTENSION)

        functional_image = load_fimg(func_file=func_file,
                                     drop_tr_count=config.PARCELLATION.TR_DROP)

        confounds = extract_confound_vars(confound_file,
                                          confounds=config.PARCELLATION.CONFOUNDS,
                                          drop_tr_count=config.PARCELLATION.TR_DROP)

        parcelled_ts = get_time_series(masker=masker,
                                       confounds=confounds,
                                       func_img=functional_image)
        logger.debug(f'Extracted time series of shape: {parcelled_ts.shape}')

        time_series.append(parcelled_ts)
        labels_list.append(masker.labels_)

    logger.info(f'Extracted time series for {len(time_series)} subjects')
    logger.info(f'Atlas ROIs: {masker.labels_}')

    return time_series, labels_list


def main():
    cfg = combine_config()

    age_groups = [
        cfg.BIDS.ADULTS_BIDS_ROOT,
        cfg.BIDS.TEENAGERS_BIDS_ROOT,
        cfg.BIDS.CHILDREN_BIDS_ROOT
    ]

    for age_group in age_groups:
        layout = get_layout(bids_root=age_group,
                            validate=cfg.BIDS.VALIDATE)

        masker = define_masker(labels_img=cfg.PARCELLATION.ATLAS_FILE,
                               standardize=cfg.PARCELLATION.STANDARDIZE,
                               memory=cfg.PARCELLATION.MEMORY,
                               detrend=cfg.PARCELLATION.DETREND,
                               high_pass=cfg.PARCELLATION.HIGH_PASS,
                               low_pass=cfg.PARCELLATION.LOW_PASS,
                               t_r=cfg.PARCELLATION.TR,
                               verbose=5)

        time_series, labels_list = extract_time_series(layout=layout,
                                                       masker=masker,
                                                       config=cfg)
        save_parcellation_results(time_series=time_series,
                                  labels_list=labels_list,
                                  out_folder=cfg.PARCELLATION.RESULTS_OUT_FOLDER,
                                  file_name=cfg.PARCELLATION.RESULTS_FILE_NAME,
                                  age_group=os.path.basename(age_group))


if __name__ == '__main__':
    main()
