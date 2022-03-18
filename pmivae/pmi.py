import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from . import ConfigVAE, VariationalAutoEncoder


class PMIVAE(BaseEstimator, TransformerMixin):

    def __init__(self, config_vae: ConfigVAE, num_samples: int):
        self._fitted = False
        self._config_vae = config_vae
        self._num_samples = num_samples
        self._binary_features = []
        self._cont_features = []
        self._imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        self._imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self._vae_model = VariationalAutoEncoder(config_vae)

    @staticmethod
    def _sampling(args):
        z_mean, z_log_var = args
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = np.random.normal(size=(batch, dim))
        return z_mean + np.exp(0.5 * z_log_var) * epsilon

    def _get_samples(self, encoded_data):
        samples = []
        for i in range(self._num_samples):
            samples.append(self._sampling(encoded_data[:2]))
        return samples

    def _apply_mi(self, vae_model, data):
        encoded_data = vae_model.encode(data)
        all_samples = self._get_samples(encoded_data)
        all_samples = np.average(all_samples, axis=0)
        return vae_model.decode(all_samples)

    def fit(self, X, y=None, **fit_params):
        if not isinstance(X, np.ndarray):
            raise TypeError("'X' must be a NumPy Array.")

        X_val = None
        if "X_val" in fit_params:
            X_val = fit_params["X_val"]
            if not isinstance(X_val, np.ndarray):
                raise TypeError("'X_val' must be a NumPy Array.")

        self._binary_features = []
        for f in range(X.shape[1]):
            X_f = X[:, f]
            if np.unique(X_f[~np.isnan(X_f)]).shape[0] <= 2:
                self._binary_features.append(f)

        self._cont_features = [i for i in range(X.shape[1]) if i not in self._binary_features]

        X_pre = X.copy()
        if np.isnan(X_pre[:, self._cont_features]).astype(int).sum() > 0:
            X_pre[:, self._cont_features] = self._imp_mean.fit_transform(X_pre[:, self._cont_features])
        if np.isnan(X_pre[:, self._binary_features]).astype(int).sum() > 0:
            X_pre[:, self._binary_features] = self._imp_mode.fit_transform(X_pre[:, self._binary_features])

        X_val_pre = None
        if X_val is not None:
            X_val_pre = X_val.copy()
            if np.isnan(X_val_pre[:, self._cont_features]).astype(int).sum() > 0:
                X_val_pre[:, self._cont_features] = self._imp_mean.fit_transform(X_val_pre[:, self._cont_features])
            if np.isnan(X_val_pre[:, self._binary_features]).astype(int).sum() > 0:
                X_val_pre[:, self._binary_features] = self._imp_mode.fit_transform(X_val_pre[:, self._binary_features])

        self._vae_model.fit(X_pre, X_pre, X_val_pre, X_val_pre)
        self._fitted = True
        return self

    def transform(self, X, y=None):
        if not self._fitted:
            raise RuntimeError("The fit method must be called before transform.")
        if not isinstance(X, np.ndarray):
            raise TypeError("'X' must be a NumPy Array.")

        X_pre = X.copy()
        if np.isnan(X_pre[:, self._cont_features]).astype(int).sum() > 0:
            X_pre[:, self._cont_features] = self._imp_mean.fit_transform(X_pre[:, self._cont_features])
        if np.isnan(X_pre[:, self._binary_features]).astype(int).sum() > 0:
            X_pre[:, self._binary_features] = self._imp_mode.fit_transform(X_pre[:, self._binary_features])
        imputed_data = self._apply_mi(self._vae_model, X_pre)

        if len(self._binary_features) > 0:
            imputed_data[:, self._binary_features] = np.around(np.clip(imputed_data[:, self._binary_features], 0, 1))
        return imputed_data
