"""
Usage example of the Partial Multiple Imputation with Variational Autoencoders (PMIVAE) with the
    Breast Cancer Wisconsin dataset. Several features are injected with Missing Completely At
    Random values. The simulated missing rate is 40%. The dataset is scaled to the range [0, 1].
    The imputation is evaluated through the Mean Absolute Error.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import numpy as np
from pmivae import ConfigVAE, PMIVAE

if __name__ == '__main__':
    data_gt = load_breast_cancer(return_X_y=True)
    data_gt = np.concatenate((data_gt[0], data_gt[1].reshape(-1, 1)), axis=1)
    data_md = data_gt.copy()
    original_shape = data_gt.shape
    num_mcar = int(round(data_gt.shape[0] * data_gt.shape[1] * 0.4))  # Missing Rate = 40%
    nan_mask = np.random.choice(data_gt.shape[0] * data_gt.shape[1], num_mcar, replace=False)
    data_md = data_md.flatten()
    data_md[nan_mask] = np.nan
    data_md = data_md.reshape(original_shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_md = scaler.fit_transform(data_md)
    data_gt = scaler.transform(data_gt)

    vae_config = ConfigVAE()
    vae_config.verbose = 0
    vae_config.epochs = 500
    vae_config.neurons = [15]
    vae_config.dropout_fc = [0.2]
    vae_config.latent_dimension = 5
    vae_config.input_shape = (original_shape[1], )

    pmivae_model = PMIVAE(vae_config, num_samples=200)
    print("[PMIVAE] Training and performing imputation...")
    data_imp = pmivae_model.fit_transform(data_md)

    mae = mean_absolute_error(data_gt.flatten()[nan_mask], data_imp.flatten()[nan_mask])
    print(f"[PMIVAE] MAE for the breast cancer wisconsin dataset: {mae:.3f}")
