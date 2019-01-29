#submission 57
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df =pd.read_csv('affairs.csv')
features = df.iloc[:,:-1]
#
features = pd.get_dummies(features, columns= ['occupation'])
features = pd.get_dummies(features, columns= ['occupation_husb'])
features = features
labels = df.iloc[:,-1]

from sklearn.model_selection import train_test_split as tts
f_train,f_test,l_train,l_test = tts(features,labels,test_size=0.25,random_state=0) 
 
#perform Classification using logistic regression
from sklearn.linear_model import LogisticRegression
lor = LogisticRegression(random_state=0)
lor.fit(f_train,l_train)
l_pred = lor.predict(f_test)

#create confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(l_test,l_pred)

#check your model accuracy using confusion matrix
cm_pred = (cm[0][0]+cm[1][1])/f_test.shape[0]
print('confusion matrix:',cm_pred*100)
#check your model accuracy using .score function()
print('score fucn:',lor.score(f_test,l_test)*100)
#calculate percentage of women who actually had an affair
affair_perc = (cm/features.shape[0])*100
print("percentage of women actually had an affair: ",affair_perc[0][0])

#predict She's a 25-year-old teacher who graduated college, 
#has been married for 3 years, has 1 child, rates herself as strongly religious, 
#rates her marriage as fair, and her husband is a farmer
print('had an Affair' if lor.predict([[3,25,3,1,4,16,0,0,0,1,0,0,0,1,0,0,0,0]])==1 else 'Never had an Affair')

#Optional Build an Optimum model
import statsmodels.formula.api as sm
features = np.append(arr = np.ones((6366,1)).astype(int), values=features, axis=1)
f_opt = features[:,:]
reg_OLS = sm.OLS(endog=labels,exog=f_opt).fit()
reg_OLS.summary()

f_opt = features[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19]]
reg_OLS = sm.OLS(endog=labels,exog=f_opt).fit()
reg_OLS.summary()

f_opt = features[:,[0,1,2,3,4,5,6,9,10,11,12,13,14,15,16,17,18,19]]
reg_OLS = sm.OLS(endog=labels,exog=f_opt).fit()
reg_OLS.summary()

f_opt = features[:,[0,1,2,3,4,6,9,10,11,12,13,14,15,16,17,18,19]]
reg_OLS = sm.OLS(endog=labels,exog=f_opt).fit()
reg_OLS.summary()

f_opt = features[:,[0,1,2,3,4,6,10,11,12,13,14,15,16,17,18,19]]
reg_OLS = sm.OLS(endog=labels,exog=f_opt).fit()
reg_OLS.summary()

f_opt = features[:,[0,1,2,3,4,6,10,12,13,14,15,16,17,18,19]]
reg_OLS = sm.OLS(endog=labels,exog=f_opt).fit()
reg_OLS.summary()

f_train,f_test,l_train,l_test = tts(f_opt,labels,test_size=0.25,random_state=0)

lor.fit(f_train,l_train)
print('score after optimisation:',lor.score(f_test,l_test)*100)



