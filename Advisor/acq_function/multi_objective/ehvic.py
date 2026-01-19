from typing import List

from scipy.stats import norm

from .ehvi import ExpectedHypervolumeImprovement


class EHVIC(ExpectedHypervolumeImprovement):
    r"""Expected Hypervolume Improvement with Constraints (minimization)."""

    def __init__(
        self,
        model: List,
        constraint_models: List,
        ref_point,
        **kwargs
    ):
        super().__init__(model=model, ref_point=ref_point, **kwargs)
        self.constraint_models = constraint_models
        self.long_name = 'Expected Hypervolume Improvement with Constraints'

    def _compute(self, X: np.ndarray, **kwargs):
        acq = super()._compute(X)
        for c_model in self.constraint_models:
            mean, var = c_model.predict(X)
            std = np.sqrt(var)
            acq *= norm.cdf(-mean / std)
        return acq
