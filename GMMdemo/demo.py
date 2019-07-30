import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
# 数据生成
X, Y = make_blobs(n_samples=200, n_features=3, centers=5, cluster_std=1.0, random_state=1)
# GMM 聚类
clf = GaussianMixture(n_components=5)
clf.fit(X)
result = clf.predict(X)

# 作图
fig1 = plt.figure(1)
ax1 = Axes3D(fig1)  # 三维空间点成图
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)

fig2 = plt.figure(2)
ax2 = Axes3D(fig2)
ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=result)
plt.show()
