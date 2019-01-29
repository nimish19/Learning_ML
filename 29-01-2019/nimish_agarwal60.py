#submission 60
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('deliveryfleet.csv')
#mean distance driven per day (Distance_feature)
#mean percentage of time a driver was >5 mph over the speed limit (speeding_feature).

#Perform K-means clustering to distinguish urban drivers and rural drivers.
features = df.iloc[:,1:].values
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2,init='k-means++',random_state=0)
Y_pred = kmeans.fit_predict(features)

plt.scatter(features[Y_pred==0,0],features[Y_pred==0,1],s=20,c='red')
plt.scatter(features[Y_pred==1,0],features[Y_pred==1,1],s=20,c='blue')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=60,c="green",label='Centroids')
plt.title('urban drivers and rural drivers')
plt.xlabel('Distance_feature')
plt.ylabel('Speed_feature')
plt.legend()
plt.show()

#Perform K-means clustering again to further distinguish speeding drivers from those who follow speed limits, 
#in addition to the rural vs. urban division.
kmeans1 = KMeans(n_clusters=4,init='k-means++',random_state=0)
Y_pred1 = kmeans1.fit_predict(features)

#Label accordingly for the 4 groups.
plt.scatter(features[Y_pred1==0,0],features[Y_pred1==0,1],s=20,c='red',label='slow and rural driver')
plt.scatter(features[Y_pred1==1,0],features[Y_pred1==1,1],s=20,c='blue',label='slow and urban driver')
plt.scatter(features[Y_pred1==2,0],features[Y_pred1==2,1],s=20,c='green',label='fast and urban driver')
plt.scatter(features[Y_pred1==3,0],features[Y_pred1==3,1],s=20,c='yellow',label='fast and rural driver')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=60,c="black",label='Centroids')
plt.title('Speeding drivers v/s SpeedLimits followers among Urban v/s Rural drivers')
plt.xlabel('Distance_feature')
plt.ylabel('Speed_feature')
plt.legend()
plt.show()