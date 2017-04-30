# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()

print (iris.feature_names) #list feature columns

print (iris.target_names) #list classified column name

print (iris.data[0]) #feature value

print (iris.target[0]) #classified value      

test_idx = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_idx)

train_data = np.delete(iris.data, test_idx, axis = 0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print (test_target)      
print(clf.predict(test_data))

      
      
  



