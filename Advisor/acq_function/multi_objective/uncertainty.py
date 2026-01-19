import numpy as np
from openbox import logger

from .base import AcquisitionFunction, SurrogateModel


class Uncertainty(AcquisitionFunction):
    r"""Half width of the confidence interval (uncertainty).

    .. math::
        \text{Uncertainty}(x) = \sqrt{\beta_t} \cdot \sigma(x)
    """

    def __init__(self, model: SurrogateModel, par: float = 1.0, **kwargs):
        super().__init__(model, **kwargs)
        self.long_name = 'Uncertainty'
        self.par = par
        self.eta = None
        self.num_data = None

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self.num_data is None:
            raise ValueError(
                'No current number of datapoints specified. Call update('
                'num_data=<int>) to inform the acquisition function '
                'about the number of datapoints.'
            )
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        mean, var = self.model.predict(X)
        std = np.sqrt(var)
        beta = 2 * np.log((X.shape[1] * self.num_data ** 2) / self.par)
        uncertainty = np.sqrt(beta) * std
        if np.any(np.isnan(uncertainty)):
            logger.warning('Uncertainty has nan-value. Set to 0.')
            uncertainty[np.isnan(uncertainty)] = 0
        return uncertainty
