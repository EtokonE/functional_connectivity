# Add the parent directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), '../../'))

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


import h5py
import json
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from src.logger import get_logger
from src.graph_analysis import GraphMetrics
from src.config.base_config import combine_config

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


logger = get_logger('RawConnectivityPrediction')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class ConnectivityMeasures:
    """Dataclass for storing functional connectivity measures"""
    pearson: Optional[np.ndarray] = None
    cross_correlation_statmodels: Optional[np.ndarray] = None
    cross_correlation_numpy: Optional[np.ndarray] = None
    coherence: Optional[np.ndarray] = None
    mutual_information: Optional[np.ndarray] = None
    euclidean_distance: Optional[np.ndarray] = None
    cityblock_distance: Optional[np.ndarray] = None
    earth_movers_distance: Optional[np.ndarray] = None
    wavelet_coherence: Optional[np.ndarray] = None
    dynamic_time_warping: Optional[np.ndarray] = None
    percentage_bend_correlation: Optional[np.ndarray] = None
    spearman_rank_correlation: Optional[np.ndarray] = None
    partial_correlation: Optional[np.ndarray] = None


@dataclass
class FunctionalConnectivity:
    """Dataclass for storing functional connectivity measures for each age group"""
    adults: ConnectivityMeasures
    teenagers: ConnectivityMeasures
    children: ConnectivityMeasures


def extract_unique_pairwise(connectivity_matrix: np.ndarray) -> np.ndarray:
    """Extract unique pairwise connectivity values from connectivity matrix
    
    Args:
        connectivity_matrix (np.ndarray): connectivity matrix (n_subjects, n_rois, n_rois)
        include_diagonal (bool): whether to include main diagonal values
    
    Returns:
        np.ndarray: unique pairwise connectivity values from connectivity matrix (n*(n-1)/2)
    """
    tril_indices = np.tril_indices(connectivity_matrix.shape[1], k = -1)
    unique_pairwise = connectivity_matrix[:, tril_indices[0], tril_indices[1]]
    return unique_pairwise


def read_group_connectivity(folder_path: str,
                            group_name: str,
                            full_matrix: bool = True) -> FunctionalConnectivity:
    """Get functional connectivity measures from h5 file
    
    Args:
        folder_path (str): path to folder with h5 files
        group_name (str): group name in h5 file. The filename should start with this name
        full_matrix (bool): whether to return full connectivity matrix or only unique pairwise values
    
    Returns:
        FunctionalConnectivity: functional connectivity measures got given group
    """
    connectivity = ConnectivityMeasures()
    for file in Path(folder_path).glob('*.h5'):
        if file.name.startswith(group_name):
            print(file.name)
            with h5py.File(file, 'r') as f:
                for measure in list(f.keys()):
                    if full_matrix:
                        setattr(connectivity, measure, f[measure][:])
                    else:
                        setattr(connectivity, measure, extract_unique_pairwise(f[measure][:]))
    return connectivity


def generate_class_labels(n_samples: int,
                          class_id) -> np.ndarray:
    """Generate class labels for given number of samples

    Args:
        n_samples (int): number of samples
        class_id (int): class id

    Returns:
        np.ndarray: class labels
    """

    return np.full(n_samples, class_id)


class SFSFeatureExtractor:
    """
    Class for performing sequential feature selection
    
    Args:
        model: model to use for feature selection
        max_features (int): maximum number of features to select
        X (np.ndarray): feature matrix
        y (np.ndarray): class labels
        cv (int): number of cross-validation folds
    """
    def __init__(self, model, max_features: int, X, y, cv: int = 5):
        self.model = model
        self.max_features = max_features
        self.X = X
        self.y = y
        self.cv = cv

    def fit_sfs(self):
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS

        sfs = SFS(
            self.model,
            k_features=self.max_features,
            forward=True,
            floating=True,
            verbose=2,
            scoring='accuracy',
            cv=self.cv,
            n_jobs=-1)

        sfs = sfs.fit(self.X, self.y)
        return sfs


    def get_best_features(self, sfs_metric_dict) -> List[str]:
        sorted_d = dict(sorted(sfs_metric_dict.items(), key=lambda item: item[1]['avg_score'], reverse=True))
        first_key = next(iter(sorted_d))
        choosen_features = sorted_d[first_key]['feature_idx']
        return choosen_features
    
    def plot_sfs_graph(self, sfs_metric_dict):
        from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
        import matplotlib.pyplot as plt

        fig1 = plot_sfs(sfs_metric_dict, kind='std_dev')
        plt.ylim([0.4, 1])
        plt.title('Sequential Forward Selection (w. StdDev)')
        plt.grid()
        plt.show()
        

    def save_sfs_graph(self, sfs_metric_dict, filename: str):
        from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
        import matplotlib.pyplot as plt

        plot_sfs(sfs_metric_dict, kind='std_dev')
        plt.ylim([0.4, 1])
        plt.title('Sequential Forward Selection (w. StdDev)')
        plt.grid()
        plt.savefig(filename)

    
    def get_metric_dict(self, sfs) -> dict:
        return sfs.get_metric_dict()
    

def save_json(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)
    

def main():
    cfg = combine_config()
    print(cfg)

    os.makedirs(cfg.ML.RAW_CONNECTIVITY_RESULTS_OUT_FOLDER, exist_ok=True)
    
    # Get connectivity measures for each age group from h5 files
    adults_connectivity = read_group_connectivity(
        cfg.CONNECTIVITY.RESULTS_OUT_FOLDER, 
        Path(cfg.BIDS.ADULTS_BIDS_ROOT).name, 
        full_matrix=False
        )
    teenagers_connectivity = read_group_connectivity(
        cfg.CONNECTIVITY.RESULTS_OUT_FOLDER, 
        Path(cfg.BIDS.TEENAGERS_BIDS_ROOT).name, 
        full_matrix=False
        )
    children_connectivity = read_group_connectivity(
        cfg.CONNECTIVITY.RESULTS_OUT_FOLDER, 
        Path(cfg.BIDS.CHILDREN_BIDS_ROOT).name, 
        full_matrix=False
        )
 

    connectivity_measures = cfg.CONNECTIVITY.MEASURES
    for measure in connectivity_measures:
        logger.info(f'Processing {measure}')
        adults = getattr(adults_connectivity, measure)
        teenagers = getattr(teenagers_connectivity, measure)
        children = getattr(children_connectivity, measure)

        y_adults = generate_class_labels(adults.shape[0], 0)
        y_teenagers = generate_class_labels(teenagers.shape[0], 1)
        y_children = generate_class_labels(children.shape[0], 2)

        assert adults is not None, logger.error(f'No {measure} for adults')
        assert teenagers is not None, logger.error(f'No {measure} for teenagers') 
        assert children is not None, logger.error(f'No {measure} for children')


        logger.info(f'Adults {measure} shape: {adults.shape}')
        logger.info(f'Teenagers {measure} shape: {teenagers.shape}')
        logger.info(f'Children {measure} shape: {children.shape}')


        X = np.concatenate((adults, teenagers, children), axis=0)
        y = np.concatenate((y_adults, y_teenagers, y_children), axis=0)
        assert X.shape[0] == y.shape[0], logger.error('X and y shapes are not equal')


        clf = SVC(
            kernel=cfg.RAW_CONNECTIVITY_SVC.KERNEL,
            C=cfg.RAW_CONNECTIVITY_SVC.C, 
            random_state=cfg.RAW_CONNECTIVITY_SVC.RANDOM_STATE
            )

        feature_extractor = SFSFeatureExtractor(
            model=clf, 
            max_features=cfg.RAW_CONNECTIVITY_FEATURE_EXTRACTOR.K_FEATURES, 
            X=X, 
            y=y, 
            cv=cfg.RAW_CONNECTIVITY_FEATURE_EXTRACTOR.CV_FOLDS
            )
        
        sfs = feature_extractor.fit_sfs()
        metric_dict = feature_extractor.get_metric_dict(sfs)
        choosen_features = feature_extractor.get_best_features(metric_dict)
        #feature_extractor.plot_sfs(metric_dict)


        clf = SVC(
            kernel=cfg.RAW_CONNECTIVITY_SVC.KERNEL,
            C=cfg.RAW_CONNECTIVITY_SVC.C, 
            random_state=cfg.RAW_CONNECTIVITY_SVC.RANDOM_STATE
            )
        
        scores = cross_val_score(
            estimator=clf, 
            X=X[:, list(choosen_features)], 
            y=y, 
            scoring=cfg.RAW_CONNECTIVITY_FEATURE_EXTRACTOR.METRIC,
            cv=cfg.RAW_CONNECTIVITY_FEATURE_EXTRACTOR.CV_FOLDS, 
            n_jobs=-1
            )
        
        logger.info(f'Average accuracy: {np.mean(scores)}')

        metric_results = {'connectivity_measure': measure, 
                          'accuracy': float(np.mean(scores)), 
                          'features': choosen_features,
                          'scores': scores,
                          'sfs_metric_dict': metric_dict}
        
        logger.info(metric_results)

        save_json(metric_results, f'{cfg.ML.RAW_CONNECTIVITY_RESULTS_OUT_FOLDER}/{measure}.json')
        feature_extractor.save_sfs_graph(metric_dict, f'{cfg.ML.RAW_CONNECTIVITY_RESULTS_OUT_FOLDER}/{measure}_sfs.png')


if __name__ == '__main__':
    main()
