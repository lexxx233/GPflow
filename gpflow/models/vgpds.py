#
# Author: Thanh Le 2018
# Github: lexxx233
# License: Apache 2.0
#
# This is implementation of the Variation Gaussian Process Dynamical Systems (VGPDS), Damianou et al 2011.
# It is provided as-is without any warranty.
#

import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import warnings

import gpflow
from gpflow.models.model import  GPModel
import gpflow.kernels as kernels
import gpflow.likelihoods as likelihoods
import gpflow.features as features
import gpflow.transforms as transforms
from gpflow import settings

from gpflow.params import Parameter
from gpflow.decors import params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.expectations import expectation
from gpflow.probability_distributions import Gaussian,DiagonalGaussian,TGaussian

class VGPDS(GPModel):
    """
    Implementation of Damianou et al 2011 - Variational Gaussian Process Dynamical Systems
    """
    def __init__(self, Y, T, kern, M=10, num_latent=3, dynamic_kern=None, Z=None, KL_weight=None):
        """
        Initialise VGPDS. This only works with Gaussian Likelihood for now

        :param Y: data matrix, size T (number of time points) x D (dimensions)
        :param T: time vector, positive real value, size 1 x T
        :param kern: Mapping kernel X -> Y specification, by default RBF
        :param M: Number of inducing points
        :param num_latent: Number of latent dimension. This is automatically found unless user force latent dimension
        :param force_latent_dim: Specify whether strict latent dimension is enforced
        :param dynamic_kern: temporal dynamics kernel specification, by default RBF
        :param Z: matrix of inducing points
        :param KL_weight: Weight of KL . weight of bound = 1 - w(KL)
        """
        X_mean = large_PCA(Y, num_latent)

        GPModel.__init__(self, X_mean, Y, kern, likelihood=likelihoods.Gaussian(), mean_function=Zero())
        del self.X  # This is a params

        self.T = np.transpose(T[np.newaxis])
        self.num_latent = num_latent
        if KL_weight is None:
            self.KL_weight = 0.5
        else:
            assert KL_weight <= 1
            assert KL_weight >= 0
            self.KL_weight = KL_weight

        #This is only one way to initialize mu_bar_q
        mu_bar_q = X_mean
        lambda_q = np.ones((self.T.shape[0], self.num_latent))

        if dynamic_kern is None:
            self.dynamic_kern = kernels.RBF(1) + kernels.Bias(1) + kernels.White(1)
        else:
            self.dynamic_kern = dynamic_kern

        self.mu_bar_q = Parameter(mu_bar_q)
        self.lambda_q = Parameter(lambda_q)

        self.num_time, self.num_latent = X_mean.shape
        self.output_dim = Y.shape[1]

        # inducing points
        if Z is None:
            # By default we initialize by subset of initial latent points
            Z = np.random.permutation(X_mean.copy())[:M]

        self.feature = features.InducingPoints(Z)

        assert len(self.feature) == M

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal likelihood.
        """
        Kt = self.dynamic_kern.K(self.T)
        iKt = tf.matrix_inverse(Kt)

        # Have to reform how we calculate expectation.
        # Reparmaterize
        Mq = tf.matmul(Kt, self.mu_bar_q) # N x Q
        Sq = tf.matrix_inverse(iKt + tf.matrix_diag(tf.transpose(tf.square(self.lambda_q)))) # Q x N x N

        qX = TGaussian(Mq, Sq)

        num_inducing = len(self.feature)

        # Compute Psi statistics
        psi0 = tf.reduce_sum(expectation(qX, self.kern))
        psi1 = expectation(qX, (self.kern, self.feature))  # N X M
        psi2 = tf.reduce_sum(expectation(qX, (self.kern, self.feature), (self.kern, self.feature)), axis=0)  # M x M

        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)  # M x M
        L = tf.cholesky(Kuu)  # K_mm ^ 1/2

        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma

        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2

        B = AAT + tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))

        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        # Compute log marginal Lower Bound
        # Which is exactly like in standard model
        # The lower bound only involves data. In essence, we are computing
        # bound \leq \int q(X)log(Y|X) dX
        D = tf.cast(tf.shape(self.Y)[1], settings.float_type)
        ND = tf.cast(tf.size(self.Y), settings.float_type)

        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.matrix_diag_part(AAT)))

        # KL[q(x) || p(x|t)]
        # Only this term involves the dynamical prior
        log_det_Kt = tf.log(tf.matrix_determinant(Kt))
        log_det_Sq = tf.log(tf.matrix_determinant(Sq))

        NQ = tf.cast(tf.size(Mq), settings.float_type)

        KL = self.num_latent * log_det_Kt + tf.reduce_sum(log_det_Sq) - NQ

        for i in range(self.num_latent):
            KL += tf.trace(tf.matmul(iKt, Sq[i, :, :]) + tf.matmul(iKt, tf.matmul(tf.expand_dims(Mq[:, i], 1),
                                                                                  tf.expand_dims(Mq[:, i], 0))))

        KL = 0.5 * KL

        return bound - KL

    def _build_predict(self, Xnew, full_cov=False):

        pass

    def _build_predict_partial(self, Xnew, Ypartial, full_cov=False):
        pass

def PCA_reduce(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.
    :param X: data array of size N (number of points) x D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q.
    """
    assert Q <= X.shape[1], 'Cannot have more latent dimensions than observed'
    evals, evecs = np.linalg.eigh(np.cov(X.T))
    W = evecs[:, -Q:]
    return (X - X.mean(0)).dot(W)

def large_PCA(X, Q):
    """
    facilitate PCA when the number of dimension is large

    :param X: data of size N x D
    :param Q: number of latent dimension. Will not be used unless 0 dimension is automatically found
    :return: PCA projection array of size N x Q
    """
    if X.shape[1] <= 3000:
        return PCA_reduce(X,Q)
    warnings.warn("Very high dimensional Data, initialize latent using Sklearn RandomizedPCA instead")
    pca = PCA(n_components=Q, svd_solver='randomized')
    return pca.fit_transform(X)

def tf_cov(x):
    '''
    Calculate covariance matrix from 2d tensor
    :param x: Input tensor N x D
    :return: D x D covariance tensor
    '''
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float64)
    cov_xx = vx - mx
    return cov_xx
