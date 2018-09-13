import numpy as np
import gpflow
import gpflow.kernels as kernels

from gpflow.models.vgpds import VGPDS
from gpflow.training.tensorflow_optimizer import _TensorFlowOptimizer
from gpflow.training.scipy_optimizer import ScipyOptimizer

import matplotlib.pyplot as plt

T = np.linspace(0, 4*np.pi, 40)

kern = kernels.RBF(2, ARD=True)

#generate some data
Y1 = np.sin(T);
Y2 = np.cos(Y1);
Y3 = Y1 - Y2
Y4 = Y1 + Y2
Y5 = np.multiply(Y1, Y2)


Y = np.stack((Y1, Y2, Y3, Y4, Y5), axis=1)

m = VGPDS(Y, T, kern, num_latent=2)

m.likelihood.variance = 0.01

print('initial parameters')
print(m.as_pandas_table())

print('optimizing parameters')
opt = _TensorFlowOptimizer()
opt.minimize(m, maxiter=100, disp=True)

print(m.as_pandas_table())
