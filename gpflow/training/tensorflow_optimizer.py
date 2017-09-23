# Copyright 2017 Artem Artemev @awav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import warnings
import tensorflow as tf

from gpflow.core.base import Build
from gpflow.core.base import GPflowError
from gpflow.training import optimizer


_REGISTERED_TENSORFLOW_OPTIMIZERS = {}


class _TensorFlowOptimizer(optimizer.Optimizer):
    def __init__(self, *args, **kwargs):
        name = self.__class__.__name__
        tf_optimizer = _get_registered_optimizer(name)
        if tf_optimizer is None:
            raise ValueError('Optimizer not found.')
        self._optimizer = tf_optimizer(*args, **kwargs)
        self._model = None
        self._minimize_operation = None
        super(_TensorFlowOptimizer, self).__init__()

    def minimize(self, *args, **kwargs):
        model = self._pop_model(kwargs)
        session = self._pop_session(model, kwargs)
        if model is None:
            mode = self.model
        if mode is None:
            raise GPflowError('Model is not specified.')
        if model and model.is_build_coherence(graph=session.graph) is Build.NO:
            raise GPflowError('Model is not built.')

        if self.minimize_operation is None:
            objective = model.objective
            self._model = model
            self._minimize_operation = self.optimizer.minimize(objective, *args, **kwargs)

        feed_dict = self._pop_feed_dict(kwargs)
        maxiter = self._pop_maxiter(kwargs)

        try:
            for _ in range(maxiter):
                session.run(self.minimize_operation, feed_dict=feed_dict)
        except KeyboardInterrupt:
            warnings.warn('Optimization interrupted.')

    @property
    def minimize_operation(self):
        return self._minimize_operation

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @model.setter
    def model(self, value):
        self._model = value
        self._minimize_operation = None


def _get_registered_optimizer(name):
    return _REGISTERED_TENSORFLOW_OPTIMIZERS.get(name)


def _register_optimizer(name, optimizer_type):
    if optimizer_type.__base__ is not tf.train.Optimizer:
        raise ValueError('Wrong TensorFlow optimizer type passed: "{0}".'
                         .format(optimizer_type))
    gp_optimizer = type(name, (_TensorFlowOptimizer, ), {})
    _REGISTERED_TENSORFLOW_OPTIMIZERS[name] = optimizer_type
    module = sys.modules[__name__]
    setattr(module, name, gp_optimizer)


# Create GPflow optimizer classes with same names as TensorFlow optimizers
for key, train_type in tf.train.__dict__.items():
    suffix = 'Optimizer'
    if key != suffix and key.endswith(suffix):
        _register_optimizer(key, train_type)


__all__ = list(_REGISTERED_TENSORFLOW_OPTIMIZERS.keys())