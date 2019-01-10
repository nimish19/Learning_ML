#Submission 47
import pandas as pd

df = pd.read_csv('Loan.csv')    
#Identify the dependent and independent variable column and split them into two different objects;
features = df.iloc[:,:-1]
labels = df.iloc[:,-1]

#Apply LabelEncoder on 'Gender', 'Married','Dependents','Education','Self_Employed','Property_Area'.
#Use .cat.codes for LabelEncoding)
features = features.drop(columns='Loan_ID')
for i in features.select_dtypes(include=[object]):
    features[i]=features[i].astype('category').cat.codes

#Apply OneHotEncoding on the ‘Property_Area’ Column.
#Use.get_dummies() class for OneHotEncoding task
features = pd.get_dummies(features,columns=['Property_Area'])

#Now Encode the dependent variable column using label encoder.
labels = labels.astype('category').cat.codes

#further split the whole dataset into test and train datasets (20% for test)
from sklearn.model_selection import train_test_split as tts
f_train,f_test,l_train,l_test = tts(features,labels,random_state=0,test_size=0.20)