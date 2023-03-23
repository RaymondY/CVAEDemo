import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN

import config

device = config.device


def cluster(prefix_index):
    path = config.data_path + '{prefix_index}.txt'.format(prefix_index=prefix_index)
    data = np.loadtxt(path, delimiter=',').astype(np.int32)
    pca = PCA(n_components=2)
    pca.fit(data)
    data_pca = pca.transform(data)
    print(data.shape)
    min_samples = np.round(data.shape[0] / 100).astype(int)
    print(min_samples)
    dbscan = DBSCAN(eps=8, min_samples=min_samples, metric="minkowski", p=0.5).fit(data)
    cluster_result = dbscan.labels_
    # show the number of clusters
    print(np.unique(cluster_result))
    # print(dbscan.components_.shape)
    # hac = AgglomerativeClustering(distance_threshold=16, metric='l1', n_clusters=None, linkage='single').fit(data)
    # cluster_result = hac.labels_
    # print(cluster_result.shape)

    # display in 3D with labels in different colors
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=cluster_result, s=0.1)
    # plt.show()

    # # display in 1D
    # plt.scatter(data_pca[:, 0], cluster_result, s=0.1)
    # plt.show()
    # display in 2D

    # filter out the noise
    data_pca = data_pca[cluster_result != -1, :]
    cluster_result = cluster_result[cluster_result != -1]
    print(data_pca.shape)
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_result, s=0.1)
    plt.show()


if __name__ == '__main__':
    cluster(3)
