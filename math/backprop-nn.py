import numpy as np
from sklearn.utils.validation import check_random_state
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class NN():
  def __init__(self, structure=[3, 3, 3], eta=0.01, n_iterations=1000, random_seed=1):
    self.structure = structure
    self.eta = eta
    self.n_iterations = n_iterations
    self.random_seed = random_seed
    self.errors = []
    self.__initLayers()
    self.__initWeights()

  def __initLayers(self):
    self.layers = []
    for i in range(len(self.structure)):
      layer = np.zeros((self.structure[i] + 1, 1))
      layer[0] = 0.0 if i == len(self.structure) - 1 else 1.0
      self.layers.append(layer)

  def __initWeights(self):
    self.W = []
    rng = check_random_state(self.random_seed)
    for i in range(1, len(self.structure)):
      self.W.append(2 * rng.random_sample((self.structure[i] + 1, self.structure[i - 1] + 1)) - 1)

  def predict(self, x):
    x = np.r_[[1.0], x]
    self.layers[0] = x

    for k in range(1, len(self.layers)):
      self.layers[k] = sigmoid(np.dot(self.W[k - 1], self.layers[k - 1]))

    return self.layers[-1]

  def fit(self, X, Y):
    D = []
    for i in range(1, len(self.structure)):
      D.append(np.zeros((self.structure[i] + 1, 1)))

    delta_W = []
    for i in range(1, len(self.structure)):
      delta_W.append(np.zeros((self.structure[i] + 1, self.structure[i - 1] + 1)))

    for _ in range(self.n_iterations):
      error = 0.0
      for x, y in zip(X, Y):
        y = np.r_[[0.0], y]
        y_hat = self.predict(x)
        diff = y_hat - y
        error += 0.5 * np.sum(diff * diff)

        # Calculate deltas for all layers
        for l, d in reversed(list(enumerate(D))):
          if l == len(D) - 1:
            # Output layer
            a_o = self.layers[-1]
            D[l] = diff * a_o * (1 - a_o)
          else:
            # Hidden layers
            k = l + 1
            a_k = self.layers[k]
            D[l] = a_k * (1 - a_k) * np.dot(self.W[k].T, D[k])

        # Calculate the weight deltas and apply them to the weights
        for i in range(len(delta_W)):
          k = i + 1
          delta_W[i] = -self.eta * np.outer(D[i], self.layers[i].T)
          self.W[i] += delta_W[i]

      self.errors.append(error)

  def plot(self):
    """ Ausgabe des Fehlers
    Die im Fehlerarray gespeicherten Fehler als Grafik ausgeben
    """
    fignr = 1
    plt.figure(fignr, figsize=(5, 5))
    plt.plot(self.errors)
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('Iteration')
    plt.ylabel('Fehler')
    plt.show()

  def dump(self):
    print("Network architecture")
    print("--------")
    print("Layers:")
    for i in range(len(self.layers)):
      print("Layer", i)
      print(self.layers[i])
    print("--------")
    print("Weights:")
    for i in range(len(self.W)):
      print("Weights from layer", i, "to", (i + 1))
      print(self.W[i])
    


def main():
  X = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
  Y = np.array([[0],    [1],    [1],    [0]])
  nn = NN(eta=0.05, n_iterations=50000, structure=[2, 2, 1])
  nn.fit(X, Y)
  nn.plot()

  print("Vorhersage für [1, 0]:", nn.predict(np.array([1, 0]))[1])

if __name__ == "__main__":
  main()
