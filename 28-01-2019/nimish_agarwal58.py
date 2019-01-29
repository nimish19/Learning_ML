#submission 58
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('mushrooms.csv')

for col in df:
    df[col] = df[col].astype('category').cat.codes


#Perform Classification on the given dataset to predict if the mushroom is edible or poisonous w.r.t. 
#it’s different attributes.
#(you can perform on habitat, population and odor as the predictors)
X = df.iloc[:,[5,21,22]]
Y = df.iloc[:,0]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split as tts
X_train, X_test, Y_train, Y_test = tts(X,Y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(X_train, Y_train)

#Check accuracy of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, knc.predict(X_test))
print('Confusion Matrix:\n',cm)
print("Accuracty:",knc.score(X_train, Y_train))
