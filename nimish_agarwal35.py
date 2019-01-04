#submission 35

import pandas as pd
#Import the training set “training_titanic.csv ”
df = pd.read_csv('training_titanic.csv') 
demo = pd.DataFrame()
# How many people in the given training set survived the disaster with the Titanic and how many of them died?
survive = (df["Survived"]).value_counts()
survive.index = ['Dead','Survivors']
demo = survive

#Calculate and print the survival rates as proportions (percentage) by setting the normalize argument to True.
survive= df["Survived"].value_counts(normalize=True)
survive.index = ['Dead_percentage','Survival_percentage']
demo=demo.append(survive*100)

#Repeat the same calculations but on subsets of survivals based on Sex:

#Males that survived vs males that passed away
male_data = df[df['Sex']=='male']
data = male_data['Survived'].value_counts()
data.index = ['Male_Dead:','Male_Survived:']
demo = demo.append(data)
#Females that survived vs Females that passed away
female_data = df[df['Sex']=='female']
data = female_data['Survived'].value_counts()
data.index = ['Female_Dead:','Female_Survived:']
demo = demo.append(data)

#Calculate and print the survival rates as proportions (percentage)
data = male_data['Survived'].value_counts(normalize=True)
data.index=['Male_dead_%','Male_survival_%']
demo = demo.append(data*100)
data = female_data['Survived'].value_counts(normalize=True)
data.index=['Female_dead_%','Female_survival_%']
demo = demo.append(data*100)
print(demo)