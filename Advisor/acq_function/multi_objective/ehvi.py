import numpy as np
from scipy.stats import norm

from .base import MultiObjectiveAcquisition


class ExpectedHypervolumeImprovement(MultiObjectiveAcquisition):
    r"""Analytical Expected Hypervolume Improvement (EHVI) for m>=2 objectives.

    This implementation assumes minimization and computes the expected
    hypervolume improvement with respect to a reference point.
    """

    def __init__(self, model, ref_point, **kwargs):
        super().__init__(model=model, ref_point=ref_point, **kwargs)
        self.long_name = 'Expected Hypervolume Improvement'
        ref_point = np.asarray(ref_point, dtype=float)
        self._cross_product_indices = np.array(
            list(np.ndindex(*([2] * ref_point.shape[0])))
        )

    def psi(self, lower: np.ndarray, upper: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        u = (upper - mu) / sigma
        return sigma * norm.pdf(u) + (mu - lower) * (1 - norm.cdf(u))

    def nu(self, lower: np.ndarray, upper: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return (upper - lower) * (1 - norm.cdf((upper - mu) / sigma))

    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self.cell_lower_bounds is None or self.cell_upper_bounds is None:
            return np.zeros((X.shape[0], 1))
        if self.cell_lower_bounds.size == 0:
            return np.zeros((X.shape[0], 1))

        num_objectives = len(self.model)
        mu = np.zeros((X.shape[0], 1, num_objectives))
        sigma = np.zeros((X.shape[0], 1, num_objectives))
        for i in range(num_objectives):
            mean, variance = self.model[i].predict(X)
            mu[:, :, i] = mean
            sigma[:, :, i] = np.sqrt(variance)

        sigma = np.maximum(sigma, 1e-12)

        psi_lu = self.psi(
            lower=self.cell_lower_bounds,
            upper=self.cell_upper_bounds,
            mu=mu,
            sigma=sigma
        )
        psi_ll = self.psi(
            lower=self.cell_lower_bounds,
            upper=self.cell_lower_bounds,
            mu=mu,
            sigma=sigma
        )
        nu = self.nu(
            lower=self.cell_lower_bounds,
            upper=self.cell_upper_bounds,
            mu=mu,
            sigma=sigma
        )
        psi_diff = psi_ll - psi_lu

        stacked_factors = np.stack([psi_diff, nu], axis=-2)

        def gather(arr, index, axis):
            data_swapped = np.swapaxes(arr, 0, axis)
            index_swapped = np.swapaxes(index, 0, axis)
            gathered = np.choose(index_swapped, data_swapped)
            return np.swapaxes(gathered, 0, axis)

        indexer = np.broadcast_to(
            self._cross_product_indices,
            stacked_factors.shape[:-2] + self._cross_product_indices.shape
        )
        all_factors_up_to_last = gather(stacked_factors, indexer, axis=-2)

        return all_factors_up_to_last.prod(axis=-1).sum(axis=-1).sum(axis=-1).reshape(-1, 1)


class EHVI(ExpectedHypervolumeImprovement):
    pass
