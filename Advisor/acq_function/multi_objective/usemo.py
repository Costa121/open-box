import random
from typing import List

import numpy as np
from platypus import NSGAII, Problem

from openbox.utils.platypus_utils import set_problem_types, get_variator
from openbox.utils.constants import MAXINT

from .base import AcquisitionFunction
from .uncertainty import Uncertainty


class USeMO(AcquisitionFunction):
    r"""Uncertainty-Aware Search for Multi-Objective Bayesian Optimization."""

    def __init__(
        self,
        model: List,
        config_space,
        random_state=1,
        acq_type='ei',
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.long_name = 'Uncertainty-Aware Search'
        self.rng = np.random.RandomState(random_state)
        random.seed(self.rng.randint(MAXINT))
        self.config_space = config_space
        self.acq_type = acq_type
        self.single_acq = None
        self.uncertainty_acq = None
        self.X = None
        self.Y = None
        self.X_dim = None
        self.Y_dim = None
        self.eta = None
        self.num_data = None
        self.uncertainties = None
        self.candidates = None

    def update(self, **kwargs):
        assert 'X' in kwargs and 'Y' in kwargs
        assert 'eta' in kwargs and 'num_data' in kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.X_dim = self.X.shape[1]
        self.Y_dim = self.Y.shape[1]
        assert self.Y_dim > 1

        from . import get_acq

        self.single_acq = [get_acq(acq_type=self.acq_type, model=m) for m in self.model]
        self.uncertainty_acq = [Uncertainty(model=m) for m in self.model]

        for i in range(self.Y_dim):
            self.single_acq[i].update(model=self.model[i], eta=self.eta[i], num_data=self.num_data)
            self.uncertainty_acq[i].update(model=self.model[i], eta=self.eta[i], num_data=self.num_data)

        def CMO(x):
            x = np.asarray(x)
            return [-self.single_acq[i](x, convert=False)[0][0] for i in range(self.Y_dim)]

        problem = Problem(self.X_dim, self.Y_dim)
        set_problem_types(self.config_space, problem)
        problem.function = CMO

        variator = get_variator(self.config_space)
        algorithm = NSGAII(problem, population_size=100, variator=variator)
        algorithm.run(2500)
        for s in algorithm.result:
            s.variables[:] = [problem.types[i].decode(s.variables[i]) for i in range(problem.nvars)]
        cheap_pareto_set = [solution.variables for solution in algorithm.result]
        cheap_pareto_set_unique = cheap_pareto_set

        single_uncertainty = np.array([
            self.uncertainty_acq[i](np.asarray(cheap_pareto_set_unique), convert=False)
            for i in range(self.Y_dim)
        ])
        single_uncertainty = single_uncertainty.reshape(self.Y_dim, -1)
        self.uncertainties = np.prod(single_uncertainty, axis=0)
        self.candidates = np.array(cheap_pareto_set_unique)

    def _compute(self, X: np.ndarray, **kwargs):
        raise NotImplementedError('Use USeMO_Maximizer to select candidates.')
