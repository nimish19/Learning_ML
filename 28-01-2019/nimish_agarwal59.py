#submission 59
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('tree_addhealth.csv')

#Build a classification tree model evaluating if an adolescent would smoke regularly or not based on: 
#gender, age, (race/ethnicity) Hispanic, White, Black, Native American and Asian, 
#alcohol use, alcohol problems, marijuana use, cocaine use, inhalant use, availability of cigarettes in the home,
#depression, and self-esteem.
df['BIO_SEX'] = df['BIO_SEX'].astype('category').cat.codes
for col in df:
    df[col] = df[col].fillna(df[col].mode()[0])
X = df.iloc[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15]]
Y = df.iloc[:,7]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split as tts
X_train,X_test,Y_train,Y_test = tts(X,Y,random_state=0,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc.fit(X_train,Y_train)
print('probability that person will smoke:',dtc.score(X_test,Y_test))

#Build a classification tree model evaluation 
#if an adolescent gets expelled or not from school based on their Gender and violent behavior
X = df.iloc[:,[0,16]]
Y = df.iloc[:,21]

X = scaler.fit_transform(X)
X_train,X_test,Y_train,Y_test = tts(X,Y,random_state=0,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy',random_state=0)
dtc.fit(X_train,Y_train)
print('probability that person will smoke:',dtc.score(X_test,Y_test))

#Use random forest in relation to regular smokers as a target and 
#explanatory variable specifically with Hispanic, White, Black, Native American and Asian.
X = df.iloc[:,1:6]
Y = df.iloc[:,7]

X_train,X_test,Y_train,Y_test = tts(X,Y,random_state=0,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion='entropy',random_state=0)
rfc.fit(X_train,Y_train)
print('probability of smoking:',rfc.score(X_test,Y_test))    
