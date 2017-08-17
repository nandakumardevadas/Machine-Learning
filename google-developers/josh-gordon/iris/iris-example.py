from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np
from IPython.display import Image  
import pydotplus
import DrawGraph as dg
iris = load_iris()
test_idx = [0, 50, 100]

trainTarget = np.delete(iris.target, test_idx)
trainData = np.delete(iris.data, test_idx, axis=0)

testTarget = iris.target[test_idx]
testData = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainData, trainTarget)

print(testTarget)

print(clf.predict(testData))
print(iris.feature_names)
print(iris.target_names)

dg.drawDecisionMakingGraph(clf, iris.feature_names, iris.target_names, "iris")
# dot_data = tree.export_graphviz(clf, out_file=None, 
#                          feature_names=iris.feature_names,  
#                          class_names=iris.target_names,  
#                          filled=True, rounded=True,  
#                          special_characters=True)  
# graph = pydotplus.graph_from_dot_data(dot_data)  
# graph.write_pdf("iris.pdf") 