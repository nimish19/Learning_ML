#Submission 55
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('PastHires.csv')

#use any technique to map Y,N to 1,0 and levels of education to some scale of 0-2
for i in df.select_dtypes(include = [object]):
    df[i] = df[i].astype('category').cat.codes
    
#Build and perform Decision tree based on the predictors 
#and see how accurate your prediction is for a being hired
features = df.iloc[:,:-1].values
labels = df.Hired.values
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=0)
dt.fit(features,labels)
print(dt.score(features, labels)*100,'%')

#Now use a random forest of 10 decision trees to predict employment of specific candidate profiles:
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 0)
rf.fit(features,labels)

#Predict employment of a currently employed 10-year veteran, previous employers 4, 
#went to top-tire school, having Bachelor's Degree without Internship.
print(rf.predict([[10,1,4,0,1,0]])*100)

#Predict employment of an unemployed 10-year veteran, ,previous employers 4, 
#didn't went to any top-tire school, having Master's Degree with Internship
print(rf.predict([[10,0,4,1,0,1]])*100) 
