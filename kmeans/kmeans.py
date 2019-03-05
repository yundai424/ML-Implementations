#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :1/19/19 1:00 PM
#@Author  :Yun Dai
#@FileName: kmeans.py

import numpy as np
import matplotlib.pyplot as plt
import sys

class Kmeans:
    def __init__(self, k, data, tolerance, plus):
        '''
        Initialize an instance of k-means
        :param k: number of clusters
        :param data: array-like data
        :param tolerance: upper bound of tolerance for convergence test
        :param plus: True for kmeans++ and False for kmeans
        '''
        self.k = k
        self.data = data
        self.n = len(data)
        self.d = len(data[0])
        self.plus = plus
        self.centroids = np.zeros((self.k, self.d))
        # one hot
        self.clusters = np.zeros(data.shape)
        self.costs = []
        self.prev_cost = sys.maxsize
        self.curr_cost = 0
        self.tolerance = tolerance


    def _init_random(self):
        #initialize the k centroids randomly for classic k-means
        indexes = np.random.uniform(0, len(self.data) - 1, self.k).astype(int)
        self.centroids = self.data[indexes]
        return

    def _init_plus(self):
        # initialize the k centroids with weighted probability for k-means++
        visited = set()
        first = np.random.randint(self.n - 1)
        visited.add(first)
        self.centroids[0] = data[first]

        for j in range(1, self.k):
            total_distances = 0
            distances = np.zeros(self.n)

            for i in range(self.n):
                distance = self._distance(data[i], self.centroids[:j]).min()
                total_distances += distance
                distances[i] = distance

            probs = distances / total_distances
            jth = np.random.choice(self.n, p=probs)
            while jth in visited:
                jth = np.random.choice(self.n, p=probs)
            visited.add(j)
            self.centroids[j] = data[jth]
        return


    def _distance(self, point, centroid):
        multiplied =  (point - centroid) * (point - centroid)
        if multiplied.ndim == 1:
            return multiplied.sum()
        else:
            return multiplied.sum(axis=1)

    def _calculate_cost(self):
        return (np.linalg.norm(self.data - (self.clusters @ self.centroids), axis=1)**2).sum()

    def _update_assignments(self):
        dsquare = -2 * self.data @ self.centroids.T + np.square(self.data).sum(axis=1)[:, None] + np.square(self.centroids).sum(axis=1)[None,:]
        new_clusters = dsquare.argmin(axis=1)
        self.clusters = (np.arange(self.k)==new_clusters[:,None]).astype(np.integer)
        return


    def _update_centroids(self):
        for i in range(self.k):
            self.centroids[i] = (np.diag(self.clusters.T[i]) @ self.data).sum(axis=0)/(self.clusters.T[i].sum())
        return


    def _update_cost(self):
        self.prev_cost = self.curr_cost
        self.curr_cost = self._calculate_cost()
        self.costs.append(self.curr_cost)
        return

    def _converge(self):
        return abs(self.curr_cost - self.prev_cost) <= self.tolerance

    def run(self):
        #main function that runs the k-means or k-means++
        if self.plus:
            self._init_plus()
        else:
            self._init_random()

        converged = False
        while not converged:
            self._update_assignments()
            self._update_centroids()
            self._update_cost()
            converged = self._converge()

        return

def plot_centroids(kmeans):
    """
    Plot the centroids in 8*8 format
    :param kmeans: kmeans instance that has already run
    :return: None
    """
    plt.gray()
    figure, ax = plt.subplots(figsize=(9, 9), ncols=3, nrows=3)
    for i in range(9):
        curr_ax = ax.reshape(-1)[i]
        curr_ax.imshow(kmeans.centroids[i].reshape(8, 8))
        curr_ax.set_xticks([])
        curr_ax.set_yticks([])
    if kmeans.plus:
        figure.savefig('./centroids_plus2.png')
        figure.suptitle('Resulting K-means++ Cluster Centers', fontsize=15)
    else:
        figure.savefig('./centroids2.png')
        figure.suptitle('Resulting K-means Cluster Centers', fontsize=15)

def plot_distortion(plus):
    figure, ax = plt.subplots(1, 1)
    for _ in range(20):
        kmeans = Kmeans(9, data, tolerance, plus)
        kmeans.run()
        ax.plot([i for i in range(len(kmeans.costs))], kmeans.costs)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    if plus:
        ax.set_title('Values of Distortion Function of K-means++')
        figure.savefig('./distortion_plus2.png')
    else:
        ax.set_title('Values of Distortion Function of K-means')
        figure.savefig('./distortion2.png')

if __name__ == '__main__':
    data = np.loadtxt('./mnist_small.txt')/255
    k = 9
    tolerance = 0.001
    kmeans = Kmeans(9, data, tolerance, False)

    kmeans.run()
    #plot the centroids
    plot_centroids(kmeans)
    #plot the value of distortion function

    plot_distortion(False)
    #now let's see how k-means++ works
    kmeanspp = Kmeans(9, data, tolerance, True)
    kmeanspp.run()
    plot_centroids(kmeanspp)
    plot_distortion(True)

