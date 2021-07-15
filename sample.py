# -*- coding: utf-8 -*-
"""
Created on Sun May 30 04:25:17 2021

@author: user
"""


import numpy as np
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

clf = DecisionTreeClassifier(criterion='gini',max_depth=3)
clf = clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

export_graphviz(clf, out_file='ttt.txt',  
                filled=True, rounded=True,
                special_characters=True,feature_names=iris.feature_names,class_names=iris.target_names)

clf = RandomForestClassifier(n_estimators=20, max_depth=3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
