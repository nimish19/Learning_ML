import pandas as pd

df = pd.read_csv("Automobile.csv")

features = df.iloc[:,:-1].values
labels = df.iloc[:,-1].values

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
imp = imp.fit(features[:,1:2])
features[:,1:2] = imp.transform(features[:,1:2])

from sklearn.model_selection import train_test_split as tts
f_train,f_test,l_train,l_test = tts(features,labels, random_state=0, test_size=0.25)
