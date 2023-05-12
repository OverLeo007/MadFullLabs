from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid

from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             homogeneity_score, completeness_score, v_measure_score,
                             silhouette_score,
                             accuracy_score)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from IPython.display import display, Markdown


def plot_pca(x, y, c_centers=None, title=None):
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x)
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y)
    plt.xlabel("Первый главный признак")
    plt.ylabel("Второй главный признак")
    if title is not None:
        plt.title(title)
    if c_centers is not None:
        c_centers = pca.transform(c_centers)
        plt.scatter(c_centers[:, 0], c_centers[:, 1],
                    c=np.arange(c_centers.shape[0]), marker="+", s=300)
    plt.show()


def count_scores(x, y, y_pred):
    return pd.Series({
        'Adjusted Rand Index': adjusted_rand_score(y, y_pred),
        'Adjusted Mutual Information': adjusted_mutual_info_score(y, y_pred),
        'Homogeneity Score': homogeneity_score(y, y_pred),
        'Completeness Score': completeness_score(y, y_pred),
        'V-Measure Score': v_measure_score(y, y_pred),
        'Silhouette Score': silhouette_score(x, y_pred),
        'Accuracy': (accuracy := accuracy_score(y, y_pred)),
        'Mistake Rate': 1 - accuracy
    })


def build_kmeans(x, y, n_clusters, title):
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(x)

    plot_pca(x, km.labels_, title=title, c_centers=km.cluster_centers_)
    display(count_scores(x, y, km.labels_))


def build_af(x, y, param_grid, title):
    fitted = []
    for params in ParameterGrid(param_grid):
        af = AffinityPropagation(**params, random_state=42).fit(x)
        fitted.append((af, silhouette_score(x, af.labels_)))

    af = max(fitted, key=lambda model: model[1])[0]
    n_clusters = len(af.cluster_centers_indices_)
    plot_pca(x, af.labels_, title=title, c_centers=af.cluster_centers_)
    print(f"Model params:\n"
          f"damping={af.damping}\n"
          f"max_iter={af.max_iter}\n"
          f"clusters={n_clusters}")
    display(count_scores(x, y, af.labels_))


def build_agg(x, y, param_grid, title):
    fitted = []
    for params in ParameterGrid(param_grid):
        agg = AgglomerativeClustering(**params).fit(x)
        fitted.append((agg, silhouette_score(x, agg.labels_)))

    agg = max(fitted, key=lambda model: model[1])[0]
    n_clusters = len(set(agg.labels_))
    plot_pca(x, agg.labels_, title=title)
    print(f"Model params:\n"
          f"linkage={agg.linkage}\n"
          f"n_clusters={n_clusters}\n"
          f"affinity={agg.affinity}")
    display(count_scores(x, y, agg.labels_))


def elbow_rule(x, title="Elbow method"):
    visualizer = KElbowVisualizer(KMeans(random_state=42),
                                  param_grid={"n_clusters": range(1, 11)},
                                  timings=False,
                                  title=title)
    visualizer.fit(x)
    visualizer.show()
    return visualizer.elbow_value_


def pipeline(x, y, df_name):
    plt.set_cmap('gist_rainbow')
    display(Markdown(f'### Визуализация {df_name} без кластеризации'))
    plot_pca(x, y, title=f"Non-classified for {df_name}")
    display(Markdown(f'### Визуализируем KMeans для {df_name} '
                     f'с использованием исходных меток классов'))
    build_kmeans(x, y,
                 cl_count := np.unique(
                     y if isinstance(y, np.ndarray) else y.values).size,
                 f"KMeans with {cl_count} classes by y-uniques for {df_name}")
    display(Markdown(f'### Правило локтя для {df_name} '))
    elbow_value = elbow_rule(x,
                             f"Elbow visualisation for {df_name}")
    display(Markdown(f'### Визуализируем KMeans для {df_name} '
                     f'с использованием кол-ва классов полученных по правилу локтя'))
    build_kmeans(x, y, elbow_value,
                 f"KMeans with {elbow_value} classes by elbow rule for {df_name}")

    display(Markdown(f'### Визуализируем AffinityPropagation для {df_name}'))
    param_grid_af = {
        'damping': [0.5, 0.6, 0.7, 0.8, 0.9],
        'max_iter': [50, 100, 200, 500],
    }
    build_af(x, y, param_grid_af,
             f"Affinity Propagation Clustering for {df_name}")

    display(Markdown(f'### Визуализируем AgglomerativeClustering для {df_name}'))
    param_grid_agg = {
        'n_clusters': [cl_count],
        'linkage': ['complete', 'average', 'single'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    }
    build_agg(x, y, param_grid_agg, f'Agglomerative Clustering by y-uniques '
                                    f'for {df_name}')
    param_grid_agg = {
        'n_clusters': [elbow_value],
        'linkage': ['complete', 'average', 'single'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    }
    build_agg(x, y, param_grid_agg, f'Agglomerative Clustering by elbow '
                                    f'for {df_name}')