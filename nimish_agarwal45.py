#Submission 45
#Categorial data
import pandas as pd

df = pd.read_csv('Automobile.csv')

#Print the data types.
print(df.dtypes)



#Build a new dataframe containing only the object columns using select_dtypes function.
obj = df.select_dtypes(include=[object])

#Find the NaN values in any column and clean them up with the most occurring value of that column. 
for i in df:
    df[i] = df[i].fillna(df[i].mode()[0])

#The ” body_style” column contains 5 different values. Perform Label Encoding like shown below format
#Label Encoding format:convertible -> 0, hardtop -> 1, hatchback -> 2, sedan -> 3, wagon -> 4
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le =LabelEncoder()
for i in obj:
    df[i] = le.fit_transform(df[i])

#Perform OneHotEncoding on “drive_wheels” column where we have values of 4wd , fwd or rwd
#Perform OneHotEncoding on ”body_style” column
ohe = OneHotEncoder(categorical_features=[7,6])
df = ohe.fit_transform(df).toarray()