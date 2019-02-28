#submission 65
import pandas as pd
df = pd.read_csv('breast_cancer.csv')

'''
Sample Code Number(id number)                     ----> represented by column A.
Clump Thickness (1 – 10)                          ----> represented by column B.
Uniformity of Cell Size(1 - 10)                   ----> represented by column C.
Uniformity of Cell Shape (1 - 10)                 ----> represented by column D.
Marginal Adhesion (1 - 10)                        ----> represented by column E.
Single Epithelial Cell Size (1 - 10)              ----> represented by column F.
Bare Nuclei (1 - 10)                              ----> represented by column G.
Bland Chromatin (1 - 10)                          ----> represented by column H.
Normal Nucleoli (1 - 10)                          ----> represented by column I.
Mitoses (1 - 10)                                  ----> represented by column J.
Class: (2 for Benign and 4 for Malignant)         ----> represented by column K. 

A Benign tumor is not a cancerous tumor and Malignant tumor is a cancerous tumor.
'''

#Impute the missing values with the most frequent values.
df['G'] = df['G'].fillna(df['G'].value_counts().index[0])
df.isnull().sum()

features = df.iloc[:,1:-1].values
labels = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split as tts
feature_train, feature_test, label_train, label_test = tts(features, labels, test_size=0.25, random_state=0)

#Perform Classification on the given data-set to predict if the tumor is cancerous or not.
from sklearn.svm import SVC
# kernel used = radial basis function
svm = SVC(kernel='rbf', random_state=0, gamma='auto')
svm.fit(feature_train, label_train)
label_pred = svm.predict(feature_test)

#predicct accuracy of model
svm.score(feature_test, label_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label_test, label_pred)

'''
Predict whether a women has Benign tumor or Malignant tumor,
if her Clump thickness is around 6, uniformity of cell size is 2, 
Uniformity of Cell Shape is 5, Marginal Adhesion is 3, Single Epithelial Cell Size is 2, 
Bare Nuclei is 7,Bland Chromatin is 9, Normal Nuclei is 2 and Mitoses is 4 
'''
prediction = svm.predict([[6,2,5,3,2,7,9,2,4]])
print('Benign tumor' if prediction[0]==2 else 'Malignant tumor')


from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=0)
new = pca.fit_transform(features)

#vishualising data
import matplotlib.pyplot as plt
plt.scatter(new[labels==2,0], new[labels==2,1], s=20, c='r')
plt.scatter(new[labels==4,0], new[labels==4,1], s=20, c='b')
plt.title('2D-converted data')
plt.xlabel('pca_component1')
plt.ylabel('pca_component2')
plt.show()

feature_train, feature_test, label_train, label_test = tts(new, labels, test_size=0.25, random_state=0)

import numpy as np
x_min, x_max = feature_train[:, 0].min() - 1, feature_train[:, 0].max() + 1
y_min, y_max = feature_train[:, 1].min() - 1, feature_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# Obtain labels for each point in mesh using the model.
svm.fit(feature_train,label_train)
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the points
plt.plot(feature_test[label_test == 2, 0], feature_test[label_test == 2, 1], 'bo', label='Class 2')
plt.plot(feature_test[label_test == 4, 0], feature_test[label_test == 4, 1], 'ro', label='Class 1')
plt.contourf(xx, yy, Z, alpha=1.0)
plt.show()