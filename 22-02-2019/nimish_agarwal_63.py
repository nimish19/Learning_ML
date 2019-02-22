#submission 63

import pandas as pd 

#read dataset crimes-data.csv
df = pd.read_csv('crime_data.csv')
features = df.iloc[:,[1,2,4]].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

#Perform dimension reduction using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
reduced_features = pca.fit_transform(features)

#group the cities using k-means based on Rape, Murder and assault predictors
from sklearn.cluster import KMeans

#using elbow method to find number of clusters in Kmeans
#wcss(Withing Cluster Sum of Squares)
wcss = []
for i in range(1,11) :
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(reduced_features)
#kmeans.inertia_ : Sum of squared distances of samples to their closest cluster center
    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt    
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()    

kmeans = KMeans(n_clusters=4)
y_kmeans = kmeans.fit_predict(reduced_features)

#cluster vishualisation
plt.scatter(reduced_features[y_kmeans==0, 0], reduced_features[y_kmeans==0, 1], s=20, c='red')
plt.scatter(reduced_features[y_kmeans==1, 0], reduced_features[y_kmeans==1, 1], s=20, c='blue')
plt.scatter(reduced_features[y_kmeans==2, 0], reduced_features[y_kmeans==2, 1], s=20, c='green')
plt.scatter(reduced_features[y_kmeans==3, 0], reduced_features[y_kmeans==3, 1], s=20, c='cyan')
#plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=30, c='black', label='Centroid')
plt.show()

#if you wish to find what the clusters signify
#df['y_kmeans'] = y_kmeans
# now find the average values in each column and compare them
#pca.components to show contribution of each features