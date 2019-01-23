#Decision Tree

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')

features = df.iloc[:,1:2].values
labels = df.Salary.values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(features, labels)

l_pred1 = regressor.predict(6.5)

feature_grid = np.arange(min(features),max(features),0.01)
feature_grid = feature_grid.reshape(-1,1)
plt.scatter(features, labels, color = 'red')
plt.plot(feature_grid, regressor.predict(feature_grid), color='blue')
plt.title('Truth or Bluff (Desicion Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()