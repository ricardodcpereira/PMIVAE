import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from . import ConfigVAE, VariationalAutoEncoder


class PMIVAE(BaseEstimator, TransformerMixin):
    """
    Implementation of the Partial Multiple Imputation with Variational Autoencoders (PMIVAE),
    according to the scikit-learn architecture: methods ``fit()``, ``transform()`` and ``fit_transform()``.

    Attributes:
        _config_vae (ConfigVAE): Data class with the configuration for the Variational Autoencoder architecture.
        _num_samples (int): Number of samples taken by the multiple imputation procedure.
        _fitted (bool): Boolean flag used to indicate if the ``fit()`` method was already invoked.
        _binary_features: List of features' indexes that are binary.
        _cont_features: List of features' indexes that are continuous.
        _imp_mean (SimpleImputer): Model used to perform the pre-imputation of continuous features with their mean.
        _imp_mode (SimpleImputer): Model used to perform the pre-imputation of binary features with their mode.
        _vae_model (VariationalAutoEncoder): Variational Autoencoder model.
    """
    def __init__(self, config_vae: ConfigVAE, num_samples: int):
        self._config_vae = config_vae
        self._num_samples = num_samples
        self._fitted = False
        self._binary_features = []
        self._cont_features = []
        self._imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        self._imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self._vae_model = VariationalAutoEncoder(config_vae)

    @staticmethod
    def _sampling(args):
        """
        Samples from the Gaussian parameters learned by the Variational Autoencoder.

        Args:
            args: Gaussian parameters.

        Returns: New sample from the distribution represented by the Gaussian parameters.

        """
        z_mean, z_log_var = args
        batch = z_mean.shape[0]
        dim = z_mean.shape[1]
        epsilon = np.random.normal(size=(batch, dim))
        return z_mean + np.exp(0.5 * z_log_var) * epsilon

    def _apply_mi(self, vae_model, data):
        """
        Applies the partial multiple imputation procedure to the Gaussian parameters learned
        by the Variational Autoencoder (VAE). After encoding the input data with the VAE,
        the Gaussian parameters are sampled ``_num_samples`` times, and the average of those
        samples is the decoding input.

        Args:
            vae_model: Variational Autoencoder model already trained.
            data: Input data to be imputed.

        Returns: The input data already decoded and imputed by the PMIVAE method.

        """
        encoded_data = vae_model.encode(data)
        all_samples = []
        for i in range(self._num_samples):
            all_samples.append(self._sampling(encoded_data[:2]))
        all_samples = np.average(all_samples, axis=0)
        return vae_model.decode(all_samples)

    def fit(self, X, y=None, **fit_params):
        """
        Fits the Variational Autoencoder (VAE) model used by the PMIVAE method.
        If the input data was not pre-imputed, the pre-imputation is performed before training the VAE.
        Continuous features are pre-imputed with their mean and binary features with their mode (categorical
        features need to be transformed into binary features before using PMIVAE).

        Args:
            X: Data used to train the Variational Autoencoder.
            y: Not applicable. This parameter only exists to maintain compatibility with the scikit-learn architecture.
            **fit_params: Can be used to supply an optional validation dataset ``X_val``.

        Returns: Instance of self.

        """
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
        """
        Performs the imputation of missing values in ``X`` using the PMIVAE method.
        If the input data was not pre-imputed, the pre-imputation is performed before training the VAE.
        Continuous features are pre-imputed with their mean and binary features with their mode (categorical
        features need to be transformed into binary features before using PMIVAE).

        Args:
            X: Data to be imputed.
            y: Not applicable. This parameter only exists to maintain compatibility with the scikit-learn architecture.

        Returns: ``X`` already imputed.

        """
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
