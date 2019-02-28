#submission 66

import pandas as pd

df = pd.read_csv('banknotes.csv')
features = df.iloc[:,1:-1].values
labels = df.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

#spliting into training and test data
from sklearn.model_selection import train_test_split as tts
features_train, features_test, labels_train, labels_test = tts(features, labels, test_size=0.20, random_state=0)

from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression(random_state=0,solver='warn')
log_classifier.fit(features_train, labels_train)

# Confusion_matrix
labels_pred = log_classifier.predict(features_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_pred, labels_test)

# Score
log_classifier.score(features_test, labels_test)
# score = 1.0

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import cross_val_score
accuracies1 = cross_val_score(estimator=log_classifier, X=features, y=labels, cv=10)
print('K-fold Score:',accuracies1.mean())
#score = 0.9702 