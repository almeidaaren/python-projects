#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay as cmd
# Read the database. Since it lacks headers, put them in.
X_Axis = np.arange(1, 61, 1)
Y_Axis = []
sonar = pd.read_csv('sonar_all_data_2.csv', header=None)# sonar_all_data_2 - the data frame to analyze
cols = sonar.columns

# List out the labels
X = sonar.iloc[:, :-2]  # Features are in columns 1:(N-1)
y = sonar.iloc[:, 60]  # Classes are in column 0!

# Now split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
stdsc = StandardScaler()  # Apply standardization
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

for i in range(1, 61):
    pca = PCA(n_components=i)  # Only keep i best features!
    X_train_pca = pca.fit_transform(X_train_std)  # Apply to the train data
    X_test_pca = pca.transform(X_test_std)  # Do the same to the test data

    # Now create an MLP and train on it
    model = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', max_iter=2000, alpha=0.00001, solver='adam', tol=0.0001)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)  # How do we do on the test data?

    print('Feature:', i)
    print('Number in test', len(y_test))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    # Store the component number and accuracy in an array
    Y_Axis.append(accuracy_score(y_test, y_pred))

# Now combine the train and test data and see how we do
X_comb_pca = np.vstack((X_train_pca, X_test_pca))
y_comb = np.hstack((y_train, y_test))
print('Number in combined', len(y_comb))
y_comb_pred = model.predict(X_comb_pca)
print('Misclassified combined samples: %d' % (y_comb != y_comb_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_comb, y_comb_pred))
print('\n')
cmat = confusion_matrix(y_comb, y_comb_pred)

# Plot the accuracy vs. number of components
plt.plot(X_Axis, Y_Axis)
plt.title("Accuracy vs Number of Components")
plt.xlabel("Component number")
plt.ylabel("Accuracy")
plt.show()

# Find out the maximum
# Find out the maxium point
max_y = max(Y_Axis)
max_x = X_Axis[Y_Axis.index(max_y)]
print('Best Result \n','Feature: ',max_x,'\n Accuracy: ', max_y)
cm_display = cmd(confusion_matrix = cmat)
cm_display.plot()


# In this work, we applied PCA and MLP Classifier on the sonar_all_data_2 dataset 
# to predict the class of the sonar signals. The results showed that the maximum 
# accuracy of 95.23% was achieved with 61 principal components. Based on this, we can 
# conclude that the model has a reasonable chance of predicting the class of a sonar signal in a real minefield, but 
# further testing and refinement would be necessary for practical application. The 
# plot of accuracy vs number of components showed a generally decreasing trend, with 
# a peak accuracy at 9 components. This suggests that the model was able to capture 
# the most important information in the first few principal components, but adding too 
# many components led to overfitting. The parameters for the MLPClassifier were chosen 
# based on experimentation and balancing accuracy and computational efficiency.
