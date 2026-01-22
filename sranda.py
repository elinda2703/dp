import numpy as np
from scipy.sparse import csr_matrix
# Example
y = np.array([2, 2, 2, 2])
W = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
], dtype=float)
n = len(y)
y_mean = y.mean()
Z = y - y_mean   # deviations from mean
S0 = W.sum()

W_sparse = csr_matrix(W)
numerator = Z @ (W_sparse @ Z)   # Z^T W Z
denominator = np.sum(Z**2)
I = (n / W_sparse.sum()) * (numerator / denominator)
print("Moran's I:", I)
