# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data #f(x) = y
Y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)


# only two lines are classifier specific
'''
from sklearn import tree
clf = tree.DecisionTreeClassifier()
'''
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

clf.fit(X_train, Y_train)

predictions = clf.predict(X_test)
print(predictions)


from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, predictions)) 
