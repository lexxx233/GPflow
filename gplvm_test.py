import gpflow
from gpflow import kernels
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(42)

data = np.load('data/three_phase_oil_flow.npz')
Y = data['Y']
labels = data['labels']

print('Number of points x Number of dimensions', Y.shape)

Q = 5
M = 20  # number of inducing pts
N = Y.shape[0]
X_mean = gpflow.models.PCA_reduce(Y, Q)  # Initialise via PCA
Z = np.random.permutation(X_mean.copy())[:M]

fHmmm = False
if (fHmmm):
    k = (kernels.RBF(3, ARD=True, active_dims=slice(0, 3)) +
         kernels.Linear(2, ARD=False, active_dims=slice(3, 5)))
else:
    k = (kernels.RBF(3, ARD=True, active_dims=[0, 1, 2]) +
         kernels.Linear(2, ARD=False, active_dims=[3, 4]))

m = gpflow.models.BayesianGPLVM(X_mean=X_mean, X_var=0.1 * np.ones((N, Q)), Y=Y,
                                kern=k, M=M, Z=Z)
m.likelihood.variance = 0.01

opt = gpflow.train.ScipyOptimizer()
m.compile()


opt.minimize(m, maxiter=gpflow.test_util.notebook_niter(100))


print(m.as_pandas_table())