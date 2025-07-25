from comet_ml import Experiment
from nilearn import image as nimg

from tqdm import tqdm
from collections.abc import Iterable
from typing import Tuple, List, Optional, Union, Dict

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_extension_files(files: list, extension='.nii.gz') -> list:
    """Leave the files with the extension"""
    return [extension_file for extension_file in files if extension in extension_file]


def pull_vox_ts(functional_image, voxel_coordinates=(1, 1, 1)):
    """Pull time series for single voxel"""
    x, y, z = voxel_coordinates
    single_voxel = functional_image.slicer[x-1:x, y-1:y, z-1:z, :].get_data()
    return single_voxel.flatten()


def plot_ts(time_series, title="Single voxel time series"):
    """Plot single voxel time series"""
    plt.figure(figsize=(16.5,5))
    plt.title(title)
    x_axis = np.arange(0, time_series.shape[0])
    plt.plot(x_axis, time_series)
    plt.xlabel('Timepoint')
    plt.ylabel('Signal Value')
    

def plot_corr_matrices(matrices, title='Pearson'):
    """Plot three correlation matrices"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5,5))
    plt.suptitle(title + " correlation connectivity matrices")
    ax1.imshow(matrices[0])
    ax2.imshow(matrices[1])
    ax3.imshow(matrices[2])
    plt.show()


def extract_confounds(confound_tsv, confounds, dt=True):
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


def initialize_experiment():
    """Inicialize comet_ml experiment"""
    import os
    from dotenv import load_dotenv
    from comet_ml import Experiment
    
    load_dotenv()

    experiment = Experiment(
        api_key=os.getenv('API_KEY'),
        project_name=os.getenv('PROJECT_NAME'),
        workspace=os.getenv('WORKSPACE'),
    )
    
    return experiment



def correct_labels(parcel_atlas):
    """Get the number of unique labels in parcellation"""
    
    atlas_labels = np.unique(parcel_atlas.get_fdata().astype(int))
    num_labels = len(atlas_labels)
    
    return num_labels


def get_sub_files(layout, sub, space='MNI152NLin2009cAsym'):
    """Get the functional and confound files for curr subject"""

    func_file = layout.get(subject=sub,
                           datatype='func', task='rest',
                           desc='preproc',
                           space=space,
                           return_type='file')
    func_file = extract_extension_files(func_file)[0]
    
    confound_file=layout.get(subject=sub, datatype='func',
                             task='rest',
                             desc='confounds',
                             return_type='file')
    confound_file = extract_extension_files(confound_file, extension='tsv')[0]
    
    return func_file, confound_file


def load_fimg(func_file, drop_tr_count=None):
    """Load functional image and drop few first TRs"""
    
    func_img = nimg.load_img(func_file)
    if drop_tr_count:
        func_img = func_img.slicer[:,:,:,drop_tr_count:]
    
    return func_img


def extract_confound_vars(confound_file, confounds, drop_tr_count=None):
    """Extract the confound variables and drop few first rows from the confound matrix"""
    
    confounds = extract_confounds(confound_file,
                                  confounds)
    if drop_tr_count:
        confounds = confounds[drop_tr_count:,:]
        
    return confounds


def get_time_series(masker, confounds, func_img):
    """
    Apply the parcellation + cleaning and fill in time series using masker
    
    returns:
        full_array - a time series for a subject that includes all regions of the atlas
        time_series - a time series for a subject that includes only regions that contains subjects voxels         
    """
    
    time_series = masker.fit_transform(func_img, confounds)
    
    #regions_kept = np.array(masker.labels_)
    #full_array = np.zeros((func_img.shape[3], num_labels))
    #full_array[:, regions_kept] = time_series
    
    return time_series #full_array, time_series


def calc_all_sub_ts(layout, params, masker):
    """
    Calculate time series for all regions and all subjects and split it int two categories: control and
    schizophrenics. 
    
    Arguments:
        layout:        BIDS object
                    
        params (dict): Experiment params. Must includes: parcel_file, confounds, tr_drop
        
        masker:        NiftiLabelsMasker
        
    Returns:
        ctrl_subjects, schz_subjects (list) : Time series for control and schizophrenics 
                                                            (can includes fill regions)
        
        ctrl_subjects_raw, schz_subjects_raw (list) : Time series for control and schizophrenics 
                                                            (dont includes fill regions)
        
        
    """
    # Lsts to store time series for all regions
    #ctrl_subjects = []
    #schz_subjects = []

    # Unfilled
    ctrl_subjects_raw = []
    schz_subjects_raw = []

    # We're going to keep track of each of our subjects labels here
    # pulled from masker.labels_
    labels_list = []
    
    # Num labels in parcell atas
    #NUM_LABELS = correct_labels(params['parcel_file'])
    
    subjects = layout.get_subjects()
    for sub in tqdm(subjects):
        # Get func file and confounds matrix
        func_file, confound_file = get_sub_files(layout, sub)
        func_img = load_fimg(func_file, drop_tr_count=params['tr_drop'])
        confounds = extract_confound_vars(confound_file, confounds=params['confounds'], 
                                          drop_tr_count=params['tr_drop'])
        
        # Get time seties for current subjects
        time_series = get_time_series(masker, confounds, 
                                      func_img)
        
        #If the subject ID starts with a "1" then they are control
        if sub.startswith('1'):
            #ctrl_subjects.append(fill_array)
            ctrl_subjects_raw.append(time_series)
        #If the subject ID starts with a "5" then they are case (case of schizophrenia)
        if sub.startswith('5'):
            #schz_subjects.append(fill_array)
            schz_subjects_raw.append(time_series)

        labels_list.append(masker.labels_)
        
    return ctrl_subjects_raw, schz_subjects_raw, labels_list


def read_h5_parcellation(parcel_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load parcellations from .h5 file

    Args:
        parcel_file: path to .h5 file with parcellations

    Returns:
        loaded_time_series: time series for all subjects
        loaded_labels_list: labels for all regions
    """
    with h5py.File(parcel_file, 'r') as f:
        loaded_time_series = f['time_series'][:]
        loaded_labels_list = f['labels_list'][:]

    return loaded_time_series, loaded_labels_list


def read_h5_connectivity(connectivity_file: str, connectivity_measures: Union[List, str]) -> Union[Dict, np.ndarray]:
    """
    Load connectivity matrices from .h5 file

    Args:
        connectivity_file (str): path to .h5 file with connectivity matrices
        connectivity_measures (Union[List, str]): list of connectivity measures to load or one connectivity measure

    Returns:
        connectivity_matrices (Union[Dict, np.ndarray]): connectivity matrices for all subjects
                        If given list of measures, returns dict with connectivity matrices for each measure
                        in format {measure: connectivity_matrices}
    """
    with h5py.File(connectivity_file, 'r') as f:
        # If we want to load only one connectivity measure
        if isinstance(connectivity_measures, str):
            return f[connectivity_measures][:]

        # If we want to load few connectivity measures
        if isinstance(connectivity_measures, Iterable):
            connectivity_matrices = {}
            for measure in connectivity_measures:
                connectivity_matrices[measure] = f[measure][:]
    return connectivity_matrices


def parcellation2list(time_series: np.ndarray) -> List[np.ndarray]:
    """
    Convert time series to list of time series for each subject

    Args:
        time_series: time series for all subjects

    Returns:
        time_series_list: list of time series for each subject
    """
    return [time_series[i, :, :] for i in range(time_series.shape[0])]


def matrix_thresholding(matrices: np.ndarray,
                        threshold: Optional[float] = None) -> np.ndarray:
    """
    Threshold matrix

    Args:
        matrices (np.ndarray): connectivity matrices (n_sub, n_regions, n_regions)
        threshold (Optional[float]): threshold value or None. If none,
                                     then threshold = one standard deviation above grand mean

    Returns:
        np.array: thresholded matrix
    """
    tresholded_matrices = matrices.copy()
    if threshold is None:
        threshold = np.mean(tresholded_matrices) + np.std(tresholded_matrices)
    for matrix in tresholded_matrices:
        np.fill_diagonal(matrix, 0)
        matrix[matrix < threshold] = 0.0
    return tresholded_matrices


def get_atlas_labels(file: str) -> np.ndarray:
    return np.genfromtxt(file, usecols=1, dtype="S", delimiter="\t", encoding=None)
