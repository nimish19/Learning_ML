#submission 44

#Import the local file cars.csv
import pandas as pd
df = pd.read_csv('cars.csv')
# don't forget to consider dependent and independent variables
features = df.iloc[:,1:].values 
labels = df.iloc[:,1].values
# split the data set equally into test set and training set
from sklearn.model_selection import train_test_split as tts
f_train,f_test,l_train,l_test = tts(features, labels, random_state = 0, test_size = 0.25)
# Print it and save all the datasets  into separate '.csv' files
print('Training Data: \n','features:\n\n',f_train,'\n\nLabels:\n\n',l_train)
print('Test Data:\nFeatures:\n\n',f_test,'\n\nLabels:\n\n',l_test)
import numpy as np
f_test = pd.DataFrame(f_test)
f_test.to_csv('N1.csv')
l_test = pd.DataFrame(l_test)
l_test.to_csv('N2.csv')
f_train = pd.DataFrame(f_train)
f_train.to_csv('N3.csv')
l_train = pd.DataFrame(l_train)
l_train.to_csv('N4.csv')
