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

from src.machine_learning.raw_connectivity_prediction import SVC, SFSFeatureExtractor, save_json
from sklearn.model_selection import cross_val_score


logger = get_logger('GraphConnectivityPrediction')


@dataclass
class FeaturesFiles:
    """
    Class to store the paths to the features files
    """
    group_files: List[Path]


@dataclass
class GraphFeatures:
    X: np.ndarray
    y: np.ndarray
    names: np.ndarray


def get_feature_files(features_folder: str,
                      feature_prefix: str = 'global'):
    adults_feature_files = FeaturesFiles(group_files=[])
    teenagers_feature_files = FeaturesFiles(group_files=[])
    children_feature_files = FeaturesFiles(group_files=[])

    for file in Path(features_folder).glob(f'*{feature_prefix}*.h5'):
        if file.name.startswith('adults'):
            adults_feature_files.group_files.append(file)
        elif file.name.startswith('yong_children'):
            children_feature_files.group_files.append(file)
        elif file.name.startswith('teenagers'):
            teenagers_feature_files.group_files.append(file)

    adults_feature_files.group_files = sorted(adults_feature_files.group_files)
    teenagers_feature_files.group_files = sorted(teenagers_feature_files.group_files)
    children_feature_files.group_files = sorted(children_feature_files.group_files)

    logger.info(f'Found {len(adults_feature_files.group_files)} adults files with prefix {feature_prefix}')
    logger.info(f'Found {len(teenagers_feature_files.group_files)} children files with prefix {feature_prefix}')
    logger.info(f'Found {len(children_feature_files.group_files)} teenagers files with prefix {feature_prefix}')

    return adults_feature_files, teenagers_feature_files, children_feature_files


def get_train_data(global_feature_files: FeaturesFiles,
                   local_feature_files: FeaturesFiles, 
                   file_id: int, 
                   class_id: int) -> GraphFeatures:
    from src.graph_analysis import GraphMetrics
    graph_metrics = GraphMetrics()
    print('process', global_feature_files.group_files[file_id])
    
    X_global, names_global = graph_metrics.load_from_h5(global_feature_files.group_files[file_id])
    y = np.full((X_global.shape[0]), class_id)

    X_local, names_local = graph_metrics.load_from_h5(local_feature_files.group_files[file_id])
    

    X = np.hstack((X_global, X_local))
    names = np.hstack((names_global, names_local))
    return GraphFeatures(X=X, y=y, names=names)


def main():
    cfg = combine_config()
    os.makedirs(cfg.ML.GRAPH_CONNECTIVITY_RESULTS_OUT_FOLDER, exist_ok=True)

    adults_feature_files_global, \
        teenagers_feature_files_global, \
            children_feature_files_global = get_feature_files(cfg.GRAPH_FEATURES.RESULTS_OUT_FOLDER,
                                                              'global')

    adults_feature_files_local, \
        teenagers_feature_files_local, \
            children_feature_files_local = get_feature_files(cfg.GRAPH_FEATURES.RESULTS_OUT_FOLDER,
                                                              'local')
    
    for i in range(len(adults_feature_files_global.group_files)):
        adults_features = get_train_data(adults_feature_files_global, 
                                        adults_feature_files_local,
                                        file_id=i, class_id=0)
        teenagers_features = get_train_data(teenagers_feature_files_global, 
                                            teenagers_feature_files_local,
                                            file_id=i, class_id=1)
        children_features = get_train_data(children_feature_files_global, 
                                        children_feature_files_local,
                                        file_id=i, class_id=1)
        
        X = np.concatenate((adults_features.X, teenagers_features.X, children_features.X), axis=0)
        y = np.concatenate((adults_features.y, teenagers_features.y, children_features.y), axis=0)

        assert X.shape[0] == y.shape[0], logger.error('X and y shapes are not equal')

        
        clf = SVC(
                kernel=cfg.GRAPH_CONNECTIVITY_SVC.KERNEL,
                C=cfg.GRAPH_CONNECTIVITY_SVC.C, 
                random_state=cfg.GRAPH_CONNECTIVITY_SVC.RANDOM_STATE
                )

        feature_extractor = SFSFeatureExtractor(
            model=clf, 
            max_features=cfg.GRAPH_CONNECTIVITY_FEATURE_EXTRACTOR.K_FEATURES, 
            X=X, 
            y=y, 
            cv=cfg.GRAPH_CONNECTIVITY_FEATURE_EXTRACTOR.CV_FOLDS
            )
        
        sfs = feature_extractor.fit_sfs()
        metric_dict = feature_extractor.get_metric_dict(sfs)
        choosen_features = feature_extractor.get_best_features(metric_dict)
        #feature_extractor.plot_sfs(metric_dict)


        clf = SVC(
            kernel=cfg.GRAPH_CONNECTIVITY_SVC.KERNEL,
            C=cfg.GRAPH_CONNECTIVITY_SVC.C, 
            random_state=cfg.GRAPH_CONNECTIVITY_SVC.RANDOM_STATE
            )
        
        scores = cross_val_score(
            estimator=clf, 
            X=X[:, list(choosen_features)], 
            y=y, 
            scoring=cfg.GRAPH_CONNECTIVITY_FEATURE_EXTRACTOR.METRIC,
            cv=cfg.GRAPH_CONNECTIVITY_FEATURE_EXTRACTOR.CV_FOLDS, 
            n_jobs=-1
            )
        
        logger.info(f'Average accuracy: {np.mean(scores)}')

        a = Path(adults_feature_files_global.group_files[i]).stem
        measure = a.partition('_global')[0][7:]

        metric_results = {'connectivity_measure': measure, 
                          'accuracy': float(np.mean(scores)), 
                          'features': choosen_features,
                          'scores': scores,
                          'sfs_metric_dict': metric_dict}
        
        logger.info(metric_results)

        save_json(metric_results, f'{cfg.ML.GRAPH_CONNECTIVITY_RESULTS_OUT_FOLDER}/{measure}.json')
        feature_extractor.save_sfs_graph(metric_dict, f'{cfg.ML.GRAPH_CONNECTIVITY_RESULTS_OUT_FOLDER}/{measure}_sfs.png')


if __name__ == '__main__':
    main()
