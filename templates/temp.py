# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import tree

iris = datasets.load_iris()
clf = DecisionTreeClassifier(criterion = 'gini')
clf.fit(iris.data, iris.target)



from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
        out_file=dot_data,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True, rounded=True,
        impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")