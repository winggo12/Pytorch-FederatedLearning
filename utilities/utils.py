import numpy as np
a = np.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
unique, counts = np.unique(a, return_counts=True)
c = dict(zip(unique, counts))
print("c: ", c)
d = dict(zip(unique, counts))
print("d: ", d)

e = c.update(d)
print("e: ", e)
