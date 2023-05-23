#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

heart_df = pd.read_csv("heart1.csv")

heart_array = heart_df.values
X = heart_array[:,0:13]
Y = heart_array[:,13]
validation_size = 0.3

seed = int(10*np.random.rand())

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,test_size=validation_size, random_state=seed)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = None, random_state = seed)
tree.fit(X_train, Y_train)

X_combined = np.vstack((X_train, X_test))
Y_combined = np.hstack((Y_train, Y_test))


## Perceptron Model
ppn = Perceptron(max_iter = 13, tol = 1e-3, eta0= 0.001, fit_intercept = True, random_state= seed, verbose= False)
ppn.fit(X_train, Y_train)
print(" ")
print("Perceptron")
print(" ")
print('Number of training dataset:', len(Y_test))
X_test_prediction = ppn.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test dataset:', round (test_data_accuracy, 4))
print('Number of testing dataset:', len(X_train))
X_train_prediction=ppn.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training dataset:', round (training_data_accuracy,4))
Y_pred = ppn.predict(X_test)
print('Misclassified samples: %d' % (Y_test != Y_pred).sum())
print('Accuracy: %.6f' % accuracy_score (Y_test, Y_pred))
Y_combined_pred = ppn.predict(X_combined)
print('Misclassified combined samples: %d' % (Y_combined != Y_combined_pred).sum())
print('Combined accuracy: %.6f' % accuracy_score (Y_combined, Y_combined_pred))
print(" ")



## Logistic Regression
lr = LogisticRegression (max_iter=100, tol=1e-4, C=1000, verbose=False, random_state=0)
lr.fit(X_train, Y_train)
print(" ")
print("Logistic Regression")
print(" ")
lr = LogisticRegression (max_iter=100, tol=1e-4, C=1000, verbose=False, random_state=0)
lr.fit(X_train, Y_train)
print('Number of training dataset:', len(Y_test))
X_test_prediction=SVM.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test dataset:', round (test_data_accuracy,4))
print('Number of testing dataset:', len(X_train))
X_train_prediction = SVM.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training dataset:', round (training_data_accuracy,4))
Y_pred = SVM.predict(X_test)
print('Misclassified samples: %d' % (Y_test != Y_pred).sum())
print('Accuracy: %.6f' % accuracy_score (Y_test, Y_pred))
Y_combined_pred = SVM.predict(X_combined)
print('Misclassified combined samples: Xd' % (Y_combined != Y_combined_pred).sum())
print('Combined accuracy: %.6f' % accuracy_score (Y_combined, Y_combined_pred))



## Support Vector Machine
SVM = SVC (kernel='linear', C = 1.0, random_state= seed, tol = 1e-3, gamma = 0.2, verbose = False)
SVM.fit(X_train, Y_train)
print("\n\n Support vector machine\n\n ")
print('Number of training dataset:', len(Y_test))
X_test_prediction = SVM.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test dataset:', round (test_data_accuracy,4))
X_train_prediction = SVM.predict(X_train)
print('Number of testing dataset:', len(X_train))
training_data_accuracy=accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training dataset:', round (training_data_accuracy,4))
Y_pred = SVM.predict(X_test)
print('Misclassified samples: %d' % (Y_test != Y_pred).sum())
print('Accuracy: %.6f' % accuracy_score (Y_test, Y_pred))
Y_combined_pred = SVM.predict(X_combined)
print('Misclassified combined samples: %d' % (Y_combined != Y_combined_pred).sum())
print('Combined accuracy: %.6f' % accuracy_score (Y_combined, Y_combined_pred))


## Decision Tree

print("\n\nDecision Tree \n\n")
print('Number of training dataset:', len(Y_test))
X_test_prediction = tree.predict(X_test)
test_data_accuracy = accuracy_score (X_test_prediction, Y_test)
print('Accuracy on test dataset:', round (test_data_accuracy,4))
print('Number of testing dataset:', len(X_train))
X_train_prediction = tree.predict(X_train)
training_data_accuracy = accuracy_score (X_train_prediction, Y_train)
print('Accuracy on training dataset:', round (training_data_accuracy,4))
Y_pred = tree.predict(X_test)
print('Misclassified samples: %d' % (Y_test != Y_pred).sum())
print('Accuracy : %.6f' % accuracy_score (Y_test, Y_pred))
Y_combined_pred = tree.predict(X_combined)
print('Misclassified combined samples: %d' % (Y_combined != Y_combined_pred).sum())
print('Combined accuracy: %.6f' % accuracy_score (Y_combined, Y_combined_pred))



## Random Forest CLassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state= seed, n_jobs = 1)
forest.fit(X_train, Y_train)
print(" ")
print("Random Forest")
print(" ")
print('Number of training dataset:', len(Y_test))
X_test_prediction= forest.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test dataset:', round (test_data_accuracy,4))
print('Number of testing dataset:', len(X_train))
X_train_prediction=forest.predict(X_train)
training_data_accuracy = accuracy_score (X_train_prediction, Y_train)
print('Accuracy on training dataset:', round (training_data_accuracy,4))
Y_pred = forest.predict(X_test)
print("Misclassified samples: %d" % (Y_test != Y_pred).sum())
print('Accuracy: %.6f' % accuracy_score(Y_test, Y_pred))
Y_combined_pred = forest.predict(X_combined)
print('Misclassified combined samples: %d' % (Y_combined != Y_combined_pred).sum())
print("Combined accuracy: %.6f" % accuracy_score(Y_combined, Y_combined_pred))



## K nearest Neighbor
knn = KNeighborsClassifier (n_neighbors = 6, p = 2, metric = 'minkowski', weights = 'distance')
knn.fit(X_train, Y_train)
print("\n\nKNN \n\n")
print('Number of training dataset:', len(Y_test))
X_test_prediction = knn.predict(X_test)
test_data_accuracy = accuracy_score (X_test_prediction, Y_test)
print('Accuracy on test dataset:', round (test_data_accuracy,4))
print('Number of testing dataset:', len(X_train))
X_train_prediction=knn.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training dataset:', round (training_data_accuracy,4))
Y_pred = knn.predict(X_test)
print("Misclassified samples: %d" % (Y_test != Y_pred).sum())
print('Accuracy: %.6f' % accuracy_score (Y_test, Y_pred))
Y_combined_pred = knn.predict(X_combined)
print("Misclassified combined samples: %d" % (Y_combined != Y_combined_pred).sum())
print('Combined accuracy: %.6f' % accuracy_score(Y_combined, Y_combined_pred))


# In[ ]:




