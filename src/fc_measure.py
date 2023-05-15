"""
Script to calculate the functional connectivity measures for the parcelled brain time series

Connectivity measures:
    - Pearson correlation
    - Cross correlation
    - Coherence
    - Wavelet coherence
    - Mutual information
    - Euclidean distance
    - Cityblock distance
    - Dynamic time warping
    - Earth mover's distance
"""

import numpy as np
from typing import List
from abc import ABC, abstractmethod


class ConnectivityMeasure(ABC):
    """
    Abstract class for connectivity measures

    Arguments:
        time_series (List[np.ndarray])

    Returns:
        correlation_matrices: np.ndarray
    """
    def __init__(self, time_series: List[np.ndarray]):
        self.time_series = time_series

    @abstractmethod
    def __call__(self):
        pass


class PearsonCorrelation(ConnectivityMeasure):
    """
    Pearson correlation

    Args:
        time_series (List[np.ndarray])

    Returns:
        correlation_matrices: np.ndarray
    """
    def __init__(self, time_series: List[np.ndarray]):
        super().__init__(time_series)

    def __call__(self):
        return self._calculate()

    def _calculate(self):
        from nilearn.connectome import ConnectivityMeasure
        from sklearn.covariance import EmpiricalCovariance

        covariance_estimator = EmpiricalCovariance()
        connectivity_correlation = ConnectivityMeasure(kind="correlation", cov_estimator=covariance_estimator)
        
        correlation_matrices = connectivity_correlation.fit_transform(self.time_series)
        return correlation_matrices


class CrossCorrelation(ConnectivityMeasure):
    """
    Cross correlation

    Args:
        time_series (List[np.ndarray])

    Returns:
        correlation_matrices: np.ndarray
    """

    def __init__(self, time_series: List[np.ndarray]):
        super().__init__(time_series)

    def __call__(self):
        return self._calculate()

    def _calculate(self):
        n_subjects = len(self.time_series)
        n_regions = self.time_series[0].shape[1]
        correlation_matrices = np.zeros((n_subjects, n_regions, n_regions))

        for i in range(n_subjects):
            pass

