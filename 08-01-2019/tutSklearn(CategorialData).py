#Categorial Data with pandas as well as sklearn
import pandas as pd

df = pd.read_csv("Data.csv")

features = df.iloc[:,:-1].values
labels = df.iloc[:,-1].values
#with sklearn
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

imp = Imputer()
features[:,1:] = imp.fit_transform(features[:,1:])

le = LabelEncoder()
features[:,0] = le.fit_transform(features[:,0])
labels = le.fit_transform(labels)

ohe = OneHotEncoder(categorical_features=[0])
features = ohe.fit_transform(features).toarray()


#with Pandas
df = df.fillna(df.mean())

for i in df.select_dtypes(include=[object]):
    df[i] = df[i].astype('category').cat.codes
    
features = df.drop("Purchased", axis=1)
labels = df["Purchased"]

features = pd.get_dummies(features,columns=["Country"])