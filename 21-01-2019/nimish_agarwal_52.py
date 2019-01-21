#submission 52
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#Import the iq_size.csv file
df = pd.read_csv('iq_size.csv')

features = df.iloc[:,1:].values
labels = df.iloc[:,0].values

#What is the IQ of an individual with a given brain size of 90, height of 70 inches, and weight 150 pounds ? 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features,labels)
pred = regressor.predict(features)
regressor.score(features,labels)
pred = regressor.predict([[90,70,150]])
print('IQ for braain size 90, height 70 and weight 150: ',pred)
#Build an optimal model and conclude which is more useful in predicting intelligence Height, Weight or brain size.
import statsmodels.formula.api as sm
features = np.append(arr = np.ones((38,1)).astype(int), values = features, axis=1)
x_features = features[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog=labels, exog=x_features).fit()
regressor_OLS.summary()
x_features = features[:,[0,1,2]]
regressor_OLS = sm.OLS(endog=labels, exog=x_features).fit()
regressor_OLS.summary()

x_features = features[:,[1,2]]
regressor_OLS = sm.OLS(endog=labels, exog=x_features).fit()
regressor_OLS.summary()

x_features = features[:,[1]]
regressor_OLS = sm.OLS(endog=labels, exog=x_features).fit()
regressor_OLS.summary()
