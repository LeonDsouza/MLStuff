# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from scipy.spatial import distance

def euc(a, b):
    return distance.euclidean(a, b)


class ScrappyKNN():
    def fit(self, X_train, Y_train):
        #randomly guess
        self.X_train = X_train
        self.Y_train = Y_train
        
    
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:    #X_test is 2d
            #label = random.choice(self.Y_train)
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist :
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]        
            
    
    
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
#from sklearn.neighbors import KNeighborsClassifier
clf = ScrappyKNN()

clf.fit(X_train, Y_train)

predictions = clf.predict(X_test)
print(predictions)


from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, predictions)) 
