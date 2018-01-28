# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def clust_points(file_name):
    """聚类"""
    img = np.array(cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2GRAY))  # 读入待聚类的图片
    new_img = np.ones(img.shape) * 255  # 新图片
    points = list()
    # 将前景像素加入列表中
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 0:
                points.append([i, j])
    # 进行dbscan聚类
    db = DBSCAN(eps=5, min_samples=10, metric='euclidean', leaf_size=20)
    if len(points) == 0:
        return new_img
    db.fit(points)
    # print db.core_sample_indices_
    # 将核心点涂色
    for i in db.core_sample_indices_:
        pos = points[i]
        new_img[pos[0], pos[1]] = 0
    return new_img


if __name__ == '__main__':
    for i in range(0, 2994):
        print(i)
        # img = clust_points('campus/img3.jpg')
        img = clust_points('overpass/img' + str(i) + '.jpg')

        cv2.imwrite('overpass_cluster1/img' + str(i) + '.jpg', img)
