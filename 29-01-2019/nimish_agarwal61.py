#submission 61
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df =pd.read_csv('tshirts.csv')

features = df.iloc[:,1:].values

#standardize the production on three sizes: small, medium, and large
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,random_state=0,init='k-means++')

Y_pred = kmeans.fit_predict(features)

plt.scatter(features[Y_pred==0,0],features[Y_pred==0,1],s=20,c='red',label='medium')
plt.scatter(features[Y_pred==1,0],features[Y_pred==1,1],s=20,c='blue',label='large')
plt.scatter(features[Y_pred==2,0],features[Y_pred==2,1],s=20,c='green',label='small')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=30,c='black',label='Centroids')
plt.title('Tshirt Sizes')
plt.xlabel('height(inches)')
plt.ylabel('weight(pounds)')
plt.legend()
plt.show()

#actual size of these 3 types of shirt to better fit your customers
print('Small size:\n',kmeans.cluster_centers_[2],kmeans.cluster_centers_[2])
print('Medium size:\n',kmeans.cluster_centers_[0],kmeans.cluster_centers_[0])
print('Large size:\n',kmeans.cluster_centers_[1],kmeans.cluster_centers_[1])