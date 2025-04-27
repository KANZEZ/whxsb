import numpy as np

a = np.array([1.0, 2.0])
lst = []

lst.append(a)
a += np.array([10.0, 10.0])

print(lst)