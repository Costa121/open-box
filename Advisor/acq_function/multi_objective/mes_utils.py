import numpy as np
from sklearn.kernel_approximation import RBFSampler


class MaxvalueEntropySearch:
    def __init__(self, model, X, Y, beta=1e6, random_state=1):
        self.model = model
        self.X = X
        self.Y = Y
        self.beta = beta
        self.rbf_features = None
        self.weights_mu = None
        self.L = None
        self.sampled_weights = None
        self.random_state = random_state

    def _get_length_scale(self) -> float:
        if hasattr(self.model, 'kernel') and hasattr(self.model.kernel, 'length_scale'):
            return float(self.model.kernel.length_scale)
        if hasattr(self.model, 'gp'):
            gp = self.model.gp
            kernel = getattr(gp, 'kernel_', None) or getattr(gp, 'kernel', None)
            if kernel is not None and hasattr(kernel, 'length_scale'):
                return float(kernel.length_scale)
        raise ValueError('Cannot determine kernel length_scale for MES.')

    def Sampling_RFM(self):
        length_scale = self._get_length_scale()
        self.rbf_features = RBFSampler(
            gamma=1 / (2 * length_scale ** 2),
            n_components=1000,
            random_state=self.random_state
        )
        X_train_features = self.rbf_features.fit_transform(np.asarray(self.X))

        A_inv = np.linalg.inv(
            (X_train_features.T).dot(X_train_features) + np.eye(self.rbf_features.n_components) / self.beta)
        self.weights_mu = A_inv.dot(X_train_features.T).dot(self.Y)
        weights_gamma = A_inv / self.beta
        self.L = np.linalg.cholesky(weights_gamma)

    def weigh_sampling(self):
        random_normal_sample = np.random.normal(0, 1, np.size(self.weights_mu))
        self.sampled_weights = np.c_[self.weights_mu] + self.L.dot(np.c_[random_normal_sample])

    def f_regression(self, x):
        X_features = self.rbf_features.fit_transform(x.reshape(1, len(x)))
        return X_features.dot(self.sampled_weights)
