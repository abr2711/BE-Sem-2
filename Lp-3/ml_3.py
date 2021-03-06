# -*- coding: utf-8 -*-
"""ML-3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W6hcTIwLN2jpIXCQ5OpT0Ex4k60hDWH2

k-NN
"""

import numpy as np

class kNN:
  def __init__(self, k):
    self.k = k
    self.X = []
    self.y = []

  def fit(self, X, y):
    self.X = self.X + X
    self.y = self.y + y

  def __distance(self, x, y):
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2

  def __get_class(self, X):
    distances = []
    for i in range(len(self.X)):
      distances.append((self.__distance(X, self.X[i]), self.y[i]))
    distances.sort()
    distances = distances[:self.k]
    counts = {}
    for d in distances:
      try: counts[d[1]] += 1
      except: counts[d[1]] = 1
    return max(counts, key = lambda i: counts[i])

  def predict(self, X):
    preds = []
    for x in X:
      preds.append(self.__get_class(x))
    return preds

  def __get_weighted_class(self, X):
    distances = []
    for i in range(len(self.X)):
      distances.append((self.__distance(X, self.X[i]), self.y[i]))
    distances.sort()
    distances = distances[:self.k]
    counts = {}
    for d in distances:
      try: counts[d[1]] += 1 / d[0]
      except: counts[d[1]] = 1 / d[0]
    return max(counts, key = lambda i: counts[i])

  def predict_weighted(self, X):
    preds = []
    for x in X:
      preds.append(self.__get_weighted_class(x))
    return preds

  def __get_locally_weighted_average_class(self, X):
    distances = []
    for i in range(len(self.X)):
      distances.append((self.__distance(X, self.X[i]), self.y[i]))
    distances.sort()
    distances = distances[:self.k]
    counts = {}
    for d in distances:
      try: counts[d[1]].append(1 / d[0])
      except: counts[d[1]] = [1 / d[0]]
    for c in counts:
      counts[c] = np.mean(counts[c])
    return max(counts, key = lambda i: counts[i])

  def predict_locally_weighted_average(self, X):
    preds = []
    for x in X:
      preds.append(self.__get_weighted_class(x))
    return preds

X = [
     (2, 4),
     (4, 6),
     (4, 4),
     (4, 2),
     (6, 4),
     (6 ,2)
]
y = ['Y', 'Y', 'B', 'Y', 'Y', 'B']

model = kNN(3)

model.fit(X, y)

print(f'Standard k-NN: {model.predict([(6, 6)])}')

print(f'Distance Weighted k-NN: {model.predict_weighted([(6, 6)])}')

print(f'Locally Weighted Average k-NN: {model.predict_locally_weighted_average([(6, 6)])}')