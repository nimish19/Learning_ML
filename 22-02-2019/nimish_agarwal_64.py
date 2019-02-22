#submission 64

from sklearn.datasets import load_iris
iris = load_iris()
iris=iris.data

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_iris = pca.fit_transform(iris)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
iris_pred = kmeans.fit_predict(reduced_iris)

#vishualize data to distinguish the 3 species of Iris flower.
import matplotlib.pyplot as plt
plt.scatter(reduced_iris[iris_pred==0, 0], reduced_iris[iris_pred==0, 1], s=20, c='red')
plt.scatter(reduced_iris[iris_pred==1, 0], reduced_iris[iris_pred==1, 1], s=20, c='blue')
plt.scatter(reduced_iris[iris_pred==2, 0], reduced_iris[iris_pred==2, 1], s=20, c='green')
plt.title('Different Species')
plt.show()