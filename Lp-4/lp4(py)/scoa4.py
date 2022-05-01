# -*- coding: utf-8 -*-
"""SCOA4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_6fe6EO4DX25FXVHMY6HIe1UDT2JxqWr
"""

import numpy as np

class NeuralNet:
  def __init__(self, input_nodes):
    self.n = input_nodes
    self.w = np.zeros((self.n))
    self.b = 0

  def __train(self, x, y):
    self.w += np.array(x) * y
    self.b += y

  def fit(self, X, y):
    assert len(X[0]) == self.n, "Invalid input shape."
    for i in range(len(X)):
      self.__train(X[i], y[i])

  def predict(self, X):
    return np.sum(self.w * X, axis = 1) + self.b

"""OR GATE"""

or_model = NeuralNet(2)

X = [
     [-1, -1],
     [-1, 1],
     [1, -1],
     [1, 1]
]
y = [-1, 1, 1, 1]

or_model.fit(X, y)

or_model.w
or_model.b

or_model.predict(X)

"""AND GATE"""

and_model = NeuralNet(2)

X = [
     [-1, -1],
     [-1, 1],
     [1, -1],
     [1, 1]
]
y = [-1, -1, -1, 1]

and_model.fit(X, y)

and_model.w
and_model.b

and_model.predict(X)

"""NOT GATE"""

not_model = NeuralNet(1)

X = [
     [-1],
     [1]
]
y = [1, -1]

not_model.fit(X, y)

not_model.w, not_model.b

not_model.predict(X)

"""NOR GATE"""

nor_model = NeuralNet(2)

X = [
     [-1, -1],
     [-1, 1],
     [1, -1],
     [1, 1]
]
y = [1, -1, -1, -1]

nor_model.fit(X, y)

nor_model.w, nor_model.b

nor_model.predict(X)

"""NAND GATE"""

nand_model = NeuralNet(2)

X = [
     [-1, -1],
     [-1, 1],
     [1, -1],
     [1, 1]
]
y = [1, 1, 1, -1]

nand_model.fit(X, y)

nand_model.predict(X)