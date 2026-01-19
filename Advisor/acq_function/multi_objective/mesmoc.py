import random
from typing import List

import numpy as np
from platypus import NSGAII, Problem

from openbox.utils.platypus_utils import set_problem_types, get_variator
from openbox.utils.constants import MAXINT

from .base import AcquisitionFunction
from .mes_utils import MaxvalueEntropySearch


class MESMOC(AcquisitionFunction):
    r"""Max-value Entropy Search for Multi-Objective Bayesian Optimization with Constraints."""

    def __init__(
        self,
        model: List,
        constraint_models: List,
        config_space,
        sample_num=1,
        random_state=1,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.long_name = 'Multi-Objective Max-value Entropy Search with Constraints'
        self.sample_num = sample_num
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        random.seed(self.rng.randint(MAXINT))
        self.config_space = config_space
        self.constraint_models = constraint_models
        self.num_constraints = len(constraint_models)
        self.constraint_perfs = None
        self.X = None
        self.Y = None
        self.X_dim = None
        self.Y_dim = None
        self.Multiplemes = None
        self.Multiplemes_constraints = None
        self.min_samples = None
        self.min_samples_constraints = None

    def update(self, **kwargs):
        assert 'X' in kwargs and 'Y' in kwargs
        assert 'constraint_perfs' in kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.X_dim = self.X.shape[1]
        self.Y_dim = self.Y.shape[1]

        self.Multiplemes = [None] * self.Y_dim
        self.Multiplemes_constraints = [None] * self.num_constraints
        for i in range(self.Y_dim):
            self.Multiplemes[i] = MaxvalueEntropySearch(
                self.model[i],
                self.X,
                self.Y[:, i],
                random_state=self.rng.randint(10000)
            )
            self.Multiplemes[i].Sampling_RFM()
        for i in range(self.num_constraints):
            self.Multiplemes_constraints[i] = MaxvalueEntropySearch(
                self.constraint_models[i],
                self.X,
                self.constraint_perfs[:, i],
                random_state=self.rng.randint(10000)
            )
            self.Multiplemes_constraints[i].Sampling_RFM()

        self.min_samples = []
        self.min_samples_constraints = []
        for _ in range(self.sample_num):
            for i in range(self.Y_dim):
                self.Multiplemes[i].weigh_sampling()
            for i in range(self.num_constraints):
                self.Multiplemes_constraints[i].weigh_sampling()

            def CMO(xi):
                xi = np.asarray(xi)
                y = [self.Multiplemes[i].f_regression(xi)[0][0] for i in range(self.Y_dim)]
                y_c = [
                    self.Multiplemes_constraints[i].f_regression(xi)[0][0]
                    for i in range(self.num_constraints)
                ]
                return y, y_c

            problem = Problem(self.X_dim, self.Y_dim, self.num_constraints)
            set_problem_types(self.config_space, problem)
            problem.constraints[:] = "<=0"
            problem.function = CMO

            variator = get_variator(self.config_space)
            algorithm = NSGAII(problem, population_size=100, variator=variator)
            algorithm.run(1500)
            cheap_pareto_front = [list(solution.objectives) for solution in algorithm.result]
            cheap_constraints_values = [list(solution.constraints) for solution in algorithm.result]
            min_of_functions = [min(f) for f in list(zip(*cheap_pareto_front))]
            min_of_constraints = [min(f) for f in list(zip(*cheap_constraints_values))]
            self.min_samples.append(min_of_functions)
            self.min_samples_constraints.append(min_of_constraints)

    def _compute(self, X: np.ndarray, **kwargs):
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        multi_obj_acq_total = np.zeros(shape=(X.shape[0], 1))
        for j in range(self.sample_num):
            multi_obj_acq_sample = np.zeros(shape=(X.shape[0], 1))
            for i in range(self.Y_dim):
                multi_obj_acq_sample += self.Multiplemes[i](X, self.min_samples[j][i])
            for i in range(self.num_constraints):
                multi_obj_acq_sample += self.Multiplemes_constraints[i](
                    X,
                    self.min_samples_constraints[j][i]
                )
            multi_obj_acq_total += multi_obj_acq_sample
        acq = multi_obj_acq_total / self.sample_num

        constraints = []
        for i in range(self.num_constraints):
            mean, _ = self.constraint_models[i].predict(X)
            constraints.append(mean)
        constraints = np.hstack(constraints)
        unsatisfied_idx = np.where(np.any(constraints > 0, axis=1, keepdims=True))
        acq[unsatisfied_idx] = -1e10
        return acq
