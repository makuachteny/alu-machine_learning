#!/usr/bin/env python3
'''
Module that performs K-means clustering
'''


import sklearn.cluster


def kmeans(X, k):
    """
    Function performs K-means clustering on a dataset.
    """
    # Perform K-means clustering
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k)
    kmeans_model.fit(X)

    # Extract the centroid means
    C = kmeans_model.cluster_centers_

    # Get the index of the cluster for each data point
    clss = kmeans_model.labels_

    return C, clss
