#submission 54
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df =pd.read_csv('bluegills.csv')
#Response variable(Dependent): length (in mm) of the fish
labels = df.length.values
#Potential Predictor (Independent Variable): age (in years) of the fish
features = df.age.values.reshape(-1,1)

#How is the length of a bluegill fish best related to its age? (Linear/Quadratic nature?)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(features,labels)
lr.score(features,labels)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)

features_poly = pf.fit_transform(features)

lr2 = LinearRegression()
lr2.fit(features_poly,labels)
lr2.score(features_poly,labels)

print('Quadratic nature' if lr2.score(features_poly,labels) > lr.score(features,labels) else 'Linear nature')

#What is the length of a randomly selected five-year-old bluegill fish? 
#Perform polynomial regression on the dataset.
print('prediction for age= 5 years: ',lr2.predict(pf.fit_transform(5)))
'''
plotting polynomial regression

features_grid = np.arange(min(features),max(features),0.1)
features_grid = features_grid.reshape((-1,1))
plt.scatter(features,labels,color = 'red')
plt.plot(features_grid,lr2.predict(pf.fit_transform(features_grid)),color ='blue')
plt.show()'''