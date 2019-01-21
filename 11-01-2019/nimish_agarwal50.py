#submission 50

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df =pd.read_csv('FoodTruck.csv') 
features = df.Population.values
labels = df.iloc[:,1].values
features = features.reshape(-1,1)
labels = labels.reshape(-1,1)

from sklearn.model_selection import train_test_split as tts
f_train,f_test,l_train,l_test = tts(features,labels,random_state=0,test_size=0.2) 
#Perform Simple Linear regression to predict the profit based on the population observed and visualize the result.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(f_train,l_train)

l_predict = regressor.predict(features)
score = regressor.score(f_test,l_test)
plt.scatter(f_train,l_train,color = 'red')
plt.scatter(f_test,l_test,color = 'blue')
plt.plot(f_train,regressor.predict(f_train),color='green')
plt.show()

#Based on the above trained results, what will be your estimated profit,
#if you set up your outlet in Jaipur? (Current population in Jaipur is 3.073 million)
print('prediction for Jaipur with population 3.073 million: \n',regressor.predict(3.073))
