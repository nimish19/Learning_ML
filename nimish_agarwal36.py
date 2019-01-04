#submission 36

#Import the training set “training_titanic.csv ”
import pandas as pd
df = pd.read_csv('training_titanic.csv') 
#fill missing age with mean of ages
df['Age'] = df['Age'].fillna(df['Age'].mean())
#create a new column Child
df['Child']=0
#Set the values of Child to 1 is the passenger's age is less than 18 years.
df["Child"][df["Age"]<18]=1

#Compare the normalized survival rates for those who are <18 and those who are older
data = df[df['Child']==1]
data = data['Survived'].value_counts(normalize=True)
data.index = ['Child_survival%','Child_dead%']
print(data)