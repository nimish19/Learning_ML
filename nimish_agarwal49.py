#submission 49
import pandas as pd

#Perform all the necessary preprocessing steps on the Red_Wine data
df = pd.read_csv('Red_Wine.csv')

df['wine names']=df['wine names'].fillna(df['wine names'].mode()[0])    #removing NaN
for i in df.iloc[:,1:]:
    df[i] = df[i].fillna(df[i].mean())

#split into features and labels    
features = df.iloc[:,:-1]
labels = df.iloc[:,-1]

#categorial data processing using pandas
features.iloc[:,0] = features.iloc[:,0].astype('category').cat.codes
labels = labels.astype('category').cat.codes
features = pd.get_dummies(features,columns=['wine names'])

#Split the data into train and test sets
from sklearn.model_selection import train_test_split as tts
f_test,f_train,l_test,l_train = tts(features,labels,random_state=0,test_size=.25)
