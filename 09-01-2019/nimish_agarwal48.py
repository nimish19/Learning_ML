#submission 48
import pandas as pd

#Import the csv file from the internet using the URL given below:
#https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
     header=None,
     usecols=[0,1,2]
    )

#Give the following column names to it 'Class label', 'Alcohol', 'Malic acid' respectively.
df.columns = ['Class label', 'Alcohol', 'Malic acid']

#The features Alcohol (percent/volume) and Malic acid (g/l) are measured on different scales.
#So, that Feature Scaling is important prior to any comparison or combination of these data.
#perform Standardization
from sklearn.preprocessing import StandardScaler 
ss = StandardScaler()
df.iloc[:,1:3] = ss.fit_transform(df.iloc[:,1:3])

# Perform Min-Max scaling using sklearn module.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df.iloc[:,1:3] = scaler.fit_transform(df.iloc[:,1:3])