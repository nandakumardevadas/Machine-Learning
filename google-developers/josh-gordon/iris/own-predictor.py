from scipy.spatial import distance
def euc(a, b):
    return distance.euclidean(a, b)

import random
class ScrappyKNN:

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        print(predictions)
        return predictions

    def closest(self, row):
        best_distance = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)) :
            distance = euc(row, self.X_train[i])
            if distance < best_distance:
                best_distance = distance
                best_index = i
        return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# from sklearn.neighbors import KNeighborsClassifier
# myClassifier = KNeighborsClassifier()
myClassifier = ScrappyKNN()
myClassifier.fit(X_train, y_train)
predictions = myClassifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))