import numpy as np
import matplotlib.pyplot as plt
import umap

z = np.random.rand(1000, 3)

print(z)
u = umap.UMAP().fit_transform(z)

plt.scatter(u[:,0], u[:,1], c=z)
plt.show()
