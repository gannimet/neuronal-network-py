import numpy as np
import numpy.linalg as nl
from collections import Counter

class MyKNN:
  """k-nearest-neighbor classifier"""

  def __init__(self, k_neighbors=3):
    self.k_neighbors = k_neighbors

  def fit(self, Xb, y=None):
    self.Xb = Xb
    self.y = y

    return self

  def predict(self, Xu, y=None):
    classIndices = []

    for i in Xu:
      distances = nl.norm(np.transpose(i - self.Xb), axis=0)
      indicesSortedDistances = np.argsort(distances)[:self.k_neighbors]
      mostFrequentClass = Counter(self.y[indicesSortedDistances]).most_common(1)[0][0]

      classIndices.append(mostFrequentClass)

    return classIndices

classifier1 = MyKNN(k_neighbors=3)
classifier2 = MyKNN(k_neighbors=5)

Xb = np.array([
  [1, 8.8], [1, 11], [1.2, 15.9], [3.7, 11], [6.1, 8.8], [9.8, 14.5], [7, 17],
  [10, 8.1], [11, 10.5], [11.8, 17.5], [16.4, 15.8]
])
y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

classifier1.fit(Xb, y)
classifier2.fit(Xb, y)

Xu = np.array([[9, 9]])

print("Classifier 1: ", classifier1.predict(Xu))
print("Classifier 2: ", classifier2.predict(Xu))