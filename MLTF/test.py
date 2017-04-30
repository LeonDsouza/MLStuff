# -*- coding: utf-8 -*-
from sklearn import tree

features = [[140, 1],[130, 1],[150, 0],[170, 0]]
labels = [0, 0, 1,1]

clf = tree.DecisionTreeClassifier() #create empty set of rules
clf = clf.fit(features, labels) #training algorithm fit into classifier object
print (clf.predict([[150,0]]))

