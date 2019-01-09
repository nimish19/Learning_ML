#Submission 46
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder  

df = pd.read_csv('Loan.csv')
df = df.drop(columns='Loan_ID')     #Don't need this column
#Identify the dependent and independent variable column and split them into two different objects;
features = df.iloc[:,:-1].values
labels = df.iloc[:,-1].values

#Apply LabelEncoder on 'Gender', 'Married','Dependents','Education','Self_Employed'.
le = LabelEncoder()
for i in range(5):
    features[:,i] = le.fit_transform(features[:,i])

#Apply Label encoding, followed by OneHotEncoding on the ‘Property_Area’ Column.
features[:,10] = le.fit_transform(features[:,10])

ohe = OneHotEncoder(categorical_features=[-1])
features = ohe.fit_transform(features).toarray()

#Now Encode the dependent variable column using label encoder.
le_label = LabelEncoder()
labels = le_label.fit_transform(labels)

#further split the whole dataset into test and train datasets (20% for test)
from sklearn.model_selection import train_test_split as tts
f_train,f_test,l_train,l_test = tts(features,labels,random_state=0,test_size=0.20)