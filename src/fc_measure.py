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
import os
import h5py
import numpy as np
from tqdm import tqdm
from typing import List
from abc import ABC, abstractmethod

from src.logger import get_logger
from src.parcellation import define_h5_path
from src.config.base_config import combine_config
from src.utils import read_h5_parcellation, parcellation2list

import warnings
warnings.filterwarnings('ignore')


log = get_logger('ConnectivityMeasure')


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
        from sklearn.covariance import LedoitWolf

        covariance_estimator = LedoitWolf()
        #covariance_estimator = EmpiricalCovariance()

        log.info('Calculating Pearson correlation')
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

        log.info('Calculating cross correlation')
        for k in tqdm(range(n_subjects)):
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

        log.info('Calculating cross correlation')
        for k in tqdm(range(n_subjects)):
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


class MutualInformation(ConnectivityMeasure):
    """Mutual information"""
    @staticmethod
    def calc_mutual_info(x, y, bins=10):
        from sklearn.metrics import mutual_info_score

        c_xy = np.histogram2d(x, y, bins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi

    @staticmethod
    def bound(x):
        return np.sqrt(1 - np.exp(-2 * x))

    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        import numpy as np

        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        log.info('Calculating mutual information')
        for k in tqdm(range(n_subjects)):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    mi = self.calc_mutual_info(time_series[k][:, i], time_series[k][:, j])
                    M[k, i, j] = self.bound(mi)

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 1)

        return M


class Coherence(ConnectivityMeasure):
    """Coherence"""
    @staticmethod
    def calc_coherence(x, y, fs=1.0, nperseg=50):
        from scipy.signal import coherence

        fs = 10e3
        f, Cxy = coherence(x, y, nperseg=nperseg)
        return Cxy.mean()

    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        log.info('Calculating coherence')
        for k in tqdm(range(n_subjects)):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    coh = self.calc_coherence(time_series[k][:, i], time_series[k][:, j])
                    M[k, i, j] = coh

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 1)

        return M


class WaveletCoherence(ConnectivityMeasure):
    """Wavelet Coherence"""
    @staticmethod
    def calc_wavelet_coherence(x, y, method='morl'):
        # Compute the CWT of the two signals
        import pywt

        cwt_x = pywt.cwt(x, scales=np.arange(1, len(x)), wavelet=method)[0]
        cwt_y = pywt.cwt(y, scales=np.arange(1, len(y)), wavelet=method)[0]

        # Compute the wavelet coherence
        return np.abs(np.sum(cwt_x * cwt_y.conj()) / np.sqrt(np.sum(cwt_x * cwt_x.conj()) * np.sum(cwt_y * cwt_y.conj())))

    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        log.info('Calculating wavelet coherence')
        for k in tqdm(range(n_subjects)):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    coh = self.calc_wavelet_coherence(time_series[k][:, i], time_series[k][:, j])
                    M[k, i, j] = coh

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 1)

        return M


class EuclideanDistance(ConnectivityMeasure):
    """Euclidean Distance"""
    @staticmethod
    def calc_euclidean_distance(x, y):
        from scipy.spatial.distance import euclidean

        return euclidean(x, y)

    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler

        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        log.info('Calculating euclidean distance')
        for k in tqdm(range(n_subjects)):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    dist = self.calc_euclidean_distance(time_series[k][:, i], time_series[k][:, j])
                    M[k, i, j] = dist

            scaler = MinMaxScaler()
            M[k] = scaler.fit_transform(M[k])

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 1)

        return M


class CityblockDistance(ConnectivityMeasure):
    """Cityblock Distance"""
    @staticmethod
    def calc_cityblock_distance(x, y):
        from scipy.spatial.distance import cityblock

        return cityblock(x, y)

    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler

        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        log.info('Calculating cityblock distance')
        for k in tqdm(range(n_subjects)):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    dist = self.calc_cityblock_distance(time_series[k][:, i], time_series[k][:, j])
                    M[k, i, j] = dist

            scaler = MinMaxScaler()
            M[k] = scaler.fit_transform(M[k])

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 0)

        return M


class EarthMoversDistance(ConnectivityMeasure):
    """Earth Mover's Distance"""
    @staticmethod
    def calc_emd(x, y):
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(x, y)

    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler

        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        log.info('Calculating earth mover\'s distance')
        for k in tqdm(range(n_subjects)):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    dist = self.calc_emd(time_series[k][:, i], time_series[k][:, j])
                    M[k, i, j] = dist

            #scaler = MinMaxScaler()
            #M[k] = scaler.fit_transform(M[k])

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 0)

        return M


class DynamicTimeWarping(ConnectivityMeasure):
    """Dynamic Time Warping"""
    @staticmethod
    def calc_dtw(x, y):
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean

        if np.ndim(x) == 1:
            x = np.expand_dims(x, axis=1)
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        distance, path = fastdtw(x, y, dist=euclidean)
        return distance

    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler

        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        log.info('Calculating dynamic time warping')
        for k in tqdm(range(n_subjects)):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    dist = self.calc_dtw(time_series[k][:, i], time_series[k][:, j])
                    M[k, i, j] = dist

            scaler = MinMaxScaler()
            M[k] = scaler.fit_transform(M[k])

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 0)

        return M
    

class PercentageBendCorrelation(ConnectivityMeasure):
    """Percentage-bend correlation"""
    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:

        def percbend(x, y, beta=.2):
            """Percentage bend correlation"""
            import numpy as np
            from scipy.stats import t
            
            X = np.column_stack((x, y))
            nx = X.shape[0]
            M = np.tile(np.median(X, axis=0), nx).reshape(X.shape)
            W = np.sort(np.abs(X - M), axis=0)
            m = int((1 - beta) * nx)
            omega = W[m - 1, :]
            P = (X - M) / omega
            P[np.isinf(P)] = 0
            P[np.isnan(P)] = 0

            # Loop over columns
            a = np.zeros((2, nx))
            for c in [0, 1]:
                psi = P[:, c]
                i1 = np.where(psi < -1)[0].size
                i2 = np.where(psi > 1)[0].size
                s = X[:, c].copy()
                s[np.where(psi < -1)[0]] = 0
                s[np.where(psi > 1)[0]] = 0
                pbos = (np.sum(s) + omega[c] * (i2 - i1)) / (s.size - i1 - i2)
                a[c] = (X[:, c] - pbos) / omega[c]

            # Bend
            a[a <= -1] = -1
            a[a >= 1] = 1

            # Get r, tval and pval
            a, b = a
            r = (a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum())
            tval = r * np.sqrt((nx - 2) / (1 - r ** 2))
            pval = 2 * t.sf(abs(tval), nx - 2)
            return r

        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        log.info('Calculating percentage-bend correlation')
        for k in tqdm(range(n_subjects)):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    corr = percbend(time_series[k][:, i], time_series[k][:, j])
                    M[k, i, j] = corr

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 1)
        return M
    

class SpearmanRankCorrelation(ConnectivityMeasure):
    """Spearman’s rank correlation coefficient"""
    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        from scipy import stats

        n_subjects = len(time_series)
        n_regions = time_series[0].shape[1]
        M = np.zeros((n_subjects, n_regions, n_regions))

        log.info('Calculating Spearman’s rank correlation')
        for k in tqdm(range(n_subjects)):
            for i in range(n_regions):
                for j in range(i, n_regions):
                    rho, _ = stats.spearmanr(time_series[k][:, i], time_series[k][:, j])
                    M[k, i, j] = rho

            M[k] = M[k] + M[k].T
            np.fill_diagonal(M[k], 1)
        return M
    

class PartialCorrelation(ConnectivityMeasure):
    """Partial correlation"""
    def calculate(self, time_series: List[np.ndarray]) -> np.ndarray:
        from nilearn.connectome import ConnectivityMeasure
        from sklearn.covariance import EmpiricalCovariance
        from sklearn.covariance import LedoitWolf

        covariance_estimator = LedoitWolf()
        #covariance_estimator = EmpiricalCovariance()

        log.info('Calculating Partial correlation')
        connectivity_correlation = ConnectivityMeasure(kind="partial correlation", cov_estimator=covariance_estimator)
        
        correlation_matrices = connectivity_correlation.fit_transform(time_series)
        return correlation_matrices


connectiviry_measures = {
    'pearson': PearsonCorrelation,
    'cross_correlation_statmodels': CrossCorrelationStatmodels,
    'cross_correlation_numpy': CrossCorrelationNumpy,
    'coherence': Coherence,
    'wavelet_coherence': WaveletCoherence,
    'mutual_information': MutualInformation,
    'euclidean_distance': EuclideanDistance,
    'cityblock_distance': CityblockDistance,
    'dynamic_time_warping': DynamicTimeWarping,
    'earth_movers_distance': EarthMoversDistance,
    'percentage_bend_correlation': PercentageBendCorrelation,
    'spearman_rank_correlation': SpearmanRankCorrelation,
    'partial_correlation': PartialCorrelation
}


def get_connectivity_measure(name: str) -> ConnectivityMeasure:
    if name not in connectiviry_measures:
        raise ValueError(f'Connectivity measure {name} not found')

    return connectiviry_measures[name]()


def calculate_connectivity(time_series: List[np.ndarray], connectivity_measure: str) -> np.ndarray:
    return get_connectivity_measure(connectivity_measure).calculate(time_series)


def process_group(parcellation_file: str,
                  connectivity_measures: List,
                  output_file: str = 'output.h5') -> None:
    """
    Calculate connectivity measures for each subject in a group
    Save the results in a .h5 file

    Args:
        parcellation_file: path to the file, contains time series for each subject after parcellation.
                           This file may be created by the script `src/parcellation.py`.
                           This file must be in hdf5 format,
                           and must contain a dataset named 'time_series' and 'labels_list'
        connectivity_measures: list of connectivity measures to calculate
        output_file: path to the output file in hdf5 format
    """
    log.info('Loading data from %s', parcellation_file)
    ts, labels = read_h5_parcellation(parcellation_file)
    ts = parcellation2list(ts)

    log.info('Calculating connectivity')
    with h5py.File(output_file, 'w') as f:
        for connectivity in connectivity_measures:
            matrices = calculate_connectivity(ts, connectivity)
            f.create_dataset(connectivity, data=matrices)

    log.info('Functional connectivity matrices saved to %s', output_file)


def main():
    cfg = combine_config()

    age_groups = [
        cfg.BIDS.ADULTS_BIDS_ROOT,
        cfg.BIDS.TEENAGERS_BIDS_ROOT,
        cfg.BIDS.CHILDREN_BIDS_ROOT
    ]

    os.makedirs(cfg.CONNECTIVITY.RESULTS_OUT_FOLDER, exist_ok=True)

    for age_group in age_groups:
        age_group_name = os.path.basename(age_group)
        log.info('Process group %s start', age_group_name)

        parcellation_path = define_h5_path(
            out_folder=cfg.PARCELLATION.RESULTS_OUT_FOLDER,
            file_name=cfg.PARCELLATION.RESULTS_FILE_NAME,
            age_group=age_group_name
        )

        connectivity_path = define_h5_path(
            out_folder=cfg.CONNECTIVITY.RESULTS_OUT_FOLDER,
            file_name=cfg.CONNECTIVITY.RESULTS_FILE_NAME,
            age_group=age_group_name
        )

        process_group(parcellation_file=parcellation_path,
                      connectivity_measures=cfg.CONNECTIVITY.MEASURES,
                      output_file=connectivity_path)

        log.info('Process group %s finished', age_group_name)


if __name__ == '__main__':
    main()
