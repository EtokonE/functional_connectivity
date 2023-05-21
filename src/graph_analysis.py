"""Implement graph metrics calculation"""
import os
import h5py
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import List, Tuple

from src.logger import get_logger
from src.parcellation import define_h5_path
from src.config.base_config import combine_config
from src.utils import read_h5_connectivity, get_atlas_labels, matrix_thresholding

import warnings
warnings.filterwarnings('ignore')


log = get_logger('Graph_analysis')


class GraphMetrics:
    def calculate_local_metrics(self,
                                connectivity_matrices: np.ndarray,
                                atlas_labels: List,
                                measure_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate local graph metrics for each subject
        Local Graph metrics included:
            - degree
            - average neighbor degree
            - betweenness centrality
            - closeness centrality
            - clustering coefficient

        Args:
            connectivity_matrices: thresholded connectivity matrices (n_sub, n_regions, n_regions)
            atlas_labels: labels for all regions in atlas

        Returns:
            X_graph: graph metrics for each subject (n_sub, n_regions * n_metrics)
            X_names: names of graph metrics (n_sub, n_metrics)
        """

        n_patients = len(connectivity_matrices)
        n_regions = len(atlas_labels)
        n_metrics = 5

        X_graph = np.zeros((n_patients, n_regions * n_metrics))
        X_names = np.empty((n_patients, n_regions * n_metrics), dtype='U150')

        for i, matr in enumerate(connectivity_matrices):
            # Build graph
            G = nx.from_numpy_array(matr)

            degree = np.array([degree[1] for degree in nx.degree(G)])
            degree_names = [(atlas_labels[i].decode('utf-8')
                             + '_degree_' + measure_name) for i in range(len(degree))]

            neighbor_degree_avg = np.array(list(nx.average_neighbor_degree(G).values()))
            neighbor_degree_avg_names = [(atlas_labels[i].decode('utf-8')
                                          + '_neighbor_degree_avg_' + measure_name) for i in range(len(degree))]

            centrality_betweenness = np.array(list(nx.betweenness_centrality(G).values()))
            centrality_betweenness_names = [(atlas_labels[i].decode('utf-8')
                                             + '_centrality_betweenness_' + measure_name) for i in range(len(degree))]

            centrality_closeness = np.array(list(nx.closeness_centrality(G).values()))
            centrality_closeness_names = [(atlas_labels[i].decode('utf-8')
                                           + '_centrality_closeness_' + measure_name) for i in range(len(degree))]

            clustering_coefficient = np.array(list(nx.clustering(G).values()))
            clustering_coefficient_names = [(atlas_labels[i].decode('utf-8')
                                             + '_clustering_coefficient_'+ measure_name) for i in range(len(degree))]

            feature_i = np.concatenate((degree,
                                        neighbor_degree_avg,
                                        centrality_betweenness,
                                        centrality_closeness,
                                        clustering_coefficient))
            names_i = np.concatenate((degree_names,
                                      neighbor_degree_avg_names,
                                      centrality_betweenness_names,
                                      centrality_closeness_names,
                                      clustering_coefficient_names))

            X_graph[i] = feature_i
            X_names[i] = names_i

        return X_graph, X_names

    @staticmethod
    def _avg_shortest_path_length(graph: nx.Graph) -> float:
        """Calculate average shortest path length graph"""
        try:
            return nx.average_shortest_path_length(graph)
        except nx.NetworkXError:
            connected_components = nx.connected_components(graph)
            largest_component = max(connected_components, key=len)

            # Create a subgraph of G consisting only of this component
            G_sub = graph.subgraph(largest_component)

            # Now calculate the average shortest path length
            return nx.average_shortest_path_length(G_sub)

    @staticmethod
    def _cost_efficiency(graph: nx.Graph) -> float:
        global_efficiency = nx.global_efficiency(graph)
        return 1 / global_efficiency

    @staticmethod
    def _radius(graph: nx.Graph) -> float:
        try:
            return nx.radius(graph)
        except nx.NetworkXError:
            connected_components = nx.connected_components(graph)
            largest_component = max(connected_components, key=len)

            # Create a subgraph of G consisting only of this component
            G_sub = graph.subgraph(largest_component)

            return nx.radius(G_sub)

    @staticmethod
    def _diameter(graph: nx.Graph) -> float:
        try:
            return nx.diameter(graph)
        except nx.NetworkXError:
            connected_components = nx.connected_components(graph)
            largest_component = max(connected_components, key=len)

            # Create a subgraph of G consisting only of this component
            G_sub = graph.subgraph(largest_component)

            return nx.diameter(G_sub)

    @staticmethod
    def _small_worldness(graph: nx.Graph) -> float:
        try:
            clustering_coefficient = nx.average_clustering(graph)
            path_length = nx.average_shortest_path_length(graph)
            random_graph = nx.gnp_random_graph(len(graph.nodes()),
                                               np.mean(list(nx.degree_histogram(graph))))
            random_clustering_coefficient = nx.average_clustering(random_graph)
            random_path_length = nx.average_shortest_path_length(random_graph)
            small_worldness = (clustering_coefficient / random_clustering_coefficient) / \
                              (path_length / random_path_length)
            return small_worldness

        except nx.NetworkXError:
            connected_components = nx.connected_components(graph)
            largest_component = max(connected_components, key=len)

            # Create a subgraph of G consisting only of this component
            G_sub = graph.subgraph(largest_component)

            clustering_coefficient = nx.average_clustering(G_sub)
            path_length = nx.average_shortest_path_length(G_sub)
            random_graph = nx.gnp_random_graph(len(G_sub.nodes()),
                                               np.mean(list(nx.degree_histogram(G_sub))))
            random_clustering_coefficient = nx.average_clustering(random_graph)
            random_path_length = nx.average_shortest_path_length(random_graph)
            small_worldness = (clustering_coefficient / random_clustering_coefficient) / (
                        path_length / random_path_length)
            return small_worldness

    def calculate_global_metrics(self,
                                 connectivity_matrices: np.ndarray,
                                 atlas_labels: List,
                                 measure_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate global metrics for each connectivity matrix in the dataset.

        Global Graph metrics included:
            - assortativity
            - average clustering coefficient
            - average shortest path length
            - cost efficiency
            - density
            - radius
            - diameter
            - transitivity
            - small worldness

        Args:
            connectivity_matrices: thresholded connectivity matrices (n_sub, n_regions, n_regions)
            atlas_labels: labels for all regions in atlas

        Returns:
            X_graph: graph metrics for each subject (n_sub, n_regions * n_metrics)
            X_names: names of graph metrics (n_sub, n_regions * n_metrics)
        """
        n_patients = len(connectivity_matrices)
        n_regions = len(atlas_labels)

        n_metrics = 9
        metric_names = np.array(['assortativity_' + measure_name,
                                 'avg_clustering_coefficient_' + measure_name,
                                 'avg_shortest_path_length_' + measure_name,
                                 'cost_efficiency_' + measure_name,
                                 'density_' + measure_name,
                                 'radius_' + measure_name,
                                 'diameter_ ' + measure_name,
                                 'transitivity_ ' + measure_name,
                                 'small_worldness_ ' + measure_name])

        assert n_metrics == len(metric_names)

        X_graph = np.zeros((n_patients, n_metrics))
        X_names = np.empty((n_patients, n_metrics), dtype='U50')

        for i, matr in enumerate(connectivity_matrices):
            # Build graph
            G = nx.from_numpy_array(matr)

            assortativity = nx.degree_assortativity_coefficient(G)
            avg_clustering_coefficient = nx.average_clustering(G)
            avg_shortest_path_length = self._avg_shortest_path_length(G)
            cost_efficiency = self._cost_efficiency(G)
            density = nx.density(G)
            radius = self._radius(G)
            diameter = self._diameter(G)
            transitivity = nx.transitivity(G)
            small_worldness = self._small_worldness(G)

            feature_i = np.array([assortativity,
                                  avg_clustering_coefficient,
                                  avg_shortest_path_length,
                                  cost_efficiency,
                                  density,
                                  radius,
                                  diameter,
                                  transitivity,
                                  small_worldness])

            X_graph[i] = feature_i
            X_names[i] = metric_names

        return X_graph, X_names

    @staticmethod
    def save_metrics(X: np.ndarray, X_names: np.ndarray, path: str) -> None:
        """
        Save metrics to h5 file

        Args:
            X (np.ndarray): graph metrics (n_patients, n_regions * n_metrics)
            X_names (np.ndarray): names of graph metrics (n_patients, n_metrics)
            path (str): path to save h5 file
        """
        with h5py.File(path, 'w') as hf:
            hf.create_dataset('X', data=X)
            hf.create_dataset('X_names', data=np.array(X_names, dtype='S'))

    @staticmethod
    def load_from_h5(path: str):
        """Load metrics from h5 file"""
        with h5py.File(path, 'r') as hf:
            X = hf['X'][:]
            X_names = hf['X_names'][:]
        return X, X_names


def main():
    cfg = combine_config()

    age_groups = [
        cfg.BIDS.ADULTS_BIDS_ROOT,
        cfg.BIDS.TEENAGERS_BIDS_ROOT,
        cfg.BIDS.CHILDREN_BIDS_ROOT
    ]

    os.makedirs(cfg.GRAPH_FEATURES.RESULTS_OUT_FOLDER, exist_ok=True)

    graph_features = GraphMetrics()

    for age_group in tqdm(age_groups):
        age_group_name = os.path.basename(age_group)
        log.info(f'Calculating graph features for group: {age_group_name}...')

        fc_measures_path = define_h5_path(
            out_folder=cfg.CONNECTIVITY.RESULTS_OUT_FOLDER,
            file_name=cfg.CONNECTIVITY.RESULTS_FILE_NAME,
            age_group=age_group_name
        )

        atlas_labels = get_atlas_labels(cfg.PARCELLATION.ATLAS_LABELS_FILE)

        for connectivity_measure in cfg.CONNECTIVITY.MEASURES:
            log.info('-' * 50)
            log.info(f'Calculating graph features for connectivity measure: {connectivity_measure}...')

            # Load connectivity matrices
            connectivity_matrices = read_h5_connectivity(
                connectivity_file=fc_measures_path,
                connectivity_measures=connectivity_measure
            )
            log.info(f'Loaded connectivity matrices from file: {fc_measures_path}, '
                     f'for measure: {connectivity_measure}, '
                     f'with shape: {connectivity_matrices.shape}')

            connectivity_matrices_thr = matrix_thresholding(connectivity_matrices)

            # Calculate local graph features
            graph_features_path_local = define_h5_path(
                out_folder=cfg.GRAPH_FEATURES.RESULTS_OUT_FOLDER,
                file_name=cfg.GRAPH_FEATURES.RESULTS_FILE_NAME,
                age_group=age_group_name,
                additional_info=connectivity_measure + '_local'
            )

            log.info(f'Calculating local graph features for connectivity measure: {connectivity_measure}...')
            X_graph_local, X_names_local = graph_features.calculate_local_metrics(
                connectivity_matrices=connectivity_matrices_thr,
                atlas_labels=list(atlas_labels),
                measure_name=connectivity_measure
            )
            log.info(f'Saving local graph features to file: {graph_features_path_local}')
            graph_features.save_metrics(X_graph_local, X_names_local, graph_features_path_local)

            # Calculate global graph features
            graph_features_path_global = define_h5_path(
                out_folder=cfg.GRAPH_FEATURES.RESULTS_OUT_FOLDER,
                file_name=cfg.GRAPH_FEATURES.RESULTS_FILE_NAME,
                age_group=age_group_name,
                additional_info=connectivity_measure + '_global'
            )

            log.info(f'Calculating global graph features for connectivity measure: {connectivity_measure}...')
            X_graph_global, X_names_global = graph_features.calculate_global_metrics(
                connectivity_matrices=connectivity_matrices_thr,
                atlas_labels=list(atlas_labels),
                measure_name=connectivity_measure
            )
            log.info(f'Saving global graph features to file: {graph_features_path_global}')
            graph_features.save_metrics(X_graph_global, X_names_global, graph_features_path_global)


if __name__ == '__main__':
    main()
