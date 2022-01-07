import numpy as np

# a = np.array([0.25, 0.5])
# b = np.array([0.75, 0.5])

# print('a + b = ', np.add(a, b)) # Vektoren addieren
# print('a * 2 = ', np.multiply(a, 2)) # Skalarmultiplikation
# print('a * b = ', np.dot(a, b)) # Skalarprodukt

c = np.array([1, 2, 3])
d = np.array([4, 5, 6])

print('c:', c)
print('d:', d)
print('c dot d:', np.dot(c.T, d))
print('c * d:', c * d)