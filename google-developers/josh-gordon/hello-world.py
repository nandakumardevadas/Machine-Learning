from sklearn import tree
from IPython.display import Image  
import pydotplus

# Array of Weight and Hardness
# 0 - Smooth 1 - Bumpy
features = [[150, 0], [160, 0], [170, 1], [190, 1]]
# 0 - Apple 1 - Orange
labels = [0, 0, 1, 1]
featureNames = ['Weight','Hardness']
targetNames = ['Apple','Orange']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[140, 1]]))

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=featureNames,  
                         class_names=targetNames,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_pdf("hello-world.pdf") 