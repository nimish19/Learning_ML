#submission 56
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

df =pd.read_csv('http://openedx.forsk.in/c4x/Forsk_Labs/ST101/asset/Auto_mpg.txt',sep=r"\s+",header=None)

#Give the column names as 
#"mpg", "cylinders", "displacement","horsepower","weight","acceleration", "model year", "origin", "car name" respectively
df.columns = ["mpg", "cylinders", "displacement","horsepower","weight","acceleration", "model year", "origin", "car name"]

#Display the Car Name with highest miles per gallon value
print(df['car name'][df['mpg']==df['mpg'].max()])

#Build the Decision Tree and Random Forest models and 
#find out which of the two is more accurate in predicting the MPG value

df['horsepower'] = df['horsepower'].replace('?',value=np.NAN)
df['horsepower']=pd.to_numeric(df['horsepower'])
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())

features = df.iloc[:,1:-1].values
labels = df.iloc[:,0].values
from sklearn.model_selection import train_test_split
f_train,f_test,l_train,l_test = train_test_split(features,labels,test_size = 0.2,random_state=0)


from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=0)
dt.fit(f_train, l_train)
print('score DecisionTree: ',dt.score(f_test,l_test))
 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0)
rf.fit(f_train, l_train)
print('score RandomForest',rf.score(f_test,l_test))
print('DecisionTree is better.' if dt.score(f_test,l_test)>rf.score(f_test,l_test) else 'RandomForest is better.')

#MPG value of a 80's model car of origin 3, weighing 2630 kgs with 6 cylinders, 
#having acceleration around 22.2 m/s due to it's 100 horsepower engine giving it a displacement of about 215. 
#(Give the prediction from both the models)

print('DecisionTree prediction:',dt.predict([[6,215,100,2630,22.2,80,3]]))
print('RandomForest prediction:',rf.predict([[6,215,100,2630,22.2,80,3]]))


