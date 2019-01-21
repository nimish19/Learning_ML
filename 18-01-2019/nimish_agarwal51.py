#submission 51
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Bahubali2_vs_Dangal.csv')

features = df.Days.values
features = features.reshape(-1,1)
labels = df.iloc[:,1:3].values.reshape(-1,2)
#predict which movie would collect more on the 10th day.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features,labels)
#label_predict = regressor.predict(features)
#score = regressor.score(features,labels)
prediction = regressor.predict(9)
