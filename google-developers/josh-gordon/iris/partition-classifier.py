from sklearn import datasets
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import DrawGraph as dg
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=.5)

# classifier = tree.DecisionTreeClassifier()
classifier = KNeighborsClassifier()
classifier = classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

# dg.drawDecisionMakingGraph(classifier, iris.feature_names, iris.target_names, "iris-partition-classifier")

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))
