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
    """Abstract class for connectivity measures"""

    @abstractmethod
    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        """
        Calculate connectivity measure

        Args:
            time_series(List[np.ndarray])

        Returns:
            correlation_matrices: np.ndarray
        """
        pass


class PearsonCorrelation(ConnectivityMeasure):
    """Pearson correlation"""
    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        from nilearn.connectome import ConnectivityMeasure
        from sklearn.covariance import EmpiricalCovariance

        covariance_estimator = EmpiricalCovariance()
        connectivity_correlation = ConnectivityMeasure(kind="correlation", cov_estimator=covariance_estimator)
        
        correlation_matrices = connectivity_correlation.fit_transform(time_series)
        return correlation_matrices


class CrossCorrelationStatmodels(ConnectivityMeasure):
    """Cross correlation with statmodels"""
    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        import statsmodels.api as sm

        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        for k in range(n_subjects):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    cross_corr = sm.tsa.stattools.ccf(time_series[k][:, i], time_series[k][:, j], adjusted=False)
                    cross_corr = cross_corr[0:(len(time_series[k][:, i]) + 1)][::-1]
                    max_cross_corr = np.max(cross_corr)
                    M[k, i, j] = max_cross_corr

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 1)
        return M


class CrossCorrelationNumpy(ConnectivityMeasure):
    """Cross correlation with numpy"""
    @staticmethod
    def _normalize_cross_correlation(cross_corr: np.ndarray,
                                     series_1: np.ndarray,
                                     series_2: np.ndarray) -> np.ndarray:
        # Compute auto-correlations of each series at zero lag
        auto_corr_1 = np.correlate(series_1, series_1, mode='valid')
        auto_corr_2 = np.correlate(series_2, series_2, mode='valid')

        # Compute normalization factor
        norm_factor = np.sqrt(auto_corr_1 * auto_corr_2)

        # Normalize cross-correlation
        normalized_cross_corr = cross_corr / norm_factor
        return normalized_cross_corr

    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        import numpy as np

        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        for k in range(n_subjects):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    cross_corr = np.correlate(time_series[k][:, i], time_series[k][:, j], mode='full')
                    normalized_cross_corr = self._normalize_cross_correlation(cross_corr,
                                                                              time_series[k][:, i],
                                                                              time_series[k][:, j])
                    max_cross_corr = np.max(abs(normalized_cross_corr))
                    M[k, i, j] = max_cross_corr

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 1)

        return M
