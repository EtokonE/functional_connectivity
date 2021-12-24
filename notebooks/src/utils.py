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
    plt.plot( x_axis, time_series)
    plt.xlabel('Timepoint')
    plt.ylabel('Signal Value')


def extract_confounds(confound_tsv, confounds, dt=True):
    """
    Extract confound matrix from tsv file
 
    Arguments:
        confound_tsv                   Full path to confounds.tsv
        confounds                      A list of confounder variables to extract
        dt                             Compute temporal derivatives [default = True]
        
    Outputs:
        confound_matrix                 
    """
    
    if dt:    
        dt_names = ['{}_derivative1'.format(c) for c in confounds]
        confounds = confounds + dt_names
    
    # Extract relevant columns
    confound_df = pd.read_csv(confound_tsv,delimiter='\t') 
    confound_df = confound_df[confounds]
    
 
    # Convert into a matrix of values (timepoints)x(variable)
    confound_mat = confound_df.values 
    
    return confound_mat
