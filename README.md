# PMIVAE - Partial Multiple Imputation with Variational Autoencoders

---
Codebase for the paper *Partial Multiple Imputation with Variational Autoencoders: Tackling Not at Randomness in Healthcare Data*

### Paper Details
- Authors: Ricardo Cardoso Pereira, Pedro Henriques Abreu, Pedro Pereira Rodrigues
- Abstract: Missing data can pose severe consequences in critical contexts, such as clinical research based on routinely collected healthcare data. This issue is usually handled with imputation strategies, but these tend to produce poor and biased results under the Missing Not At Random (MNAR) mechanism. A recent trend that has been showing promising results for MNAR is the use of generative models, particularly Variational Autoencoders. However, they have a limitation: the imputed values are the result of a single sample, which can be biased. To tackle it, an extension to the Variational Autoencoder that uses a partial multiple imputation procedure is introduced in this work. The proposed method was compared to 8 state-of-the-art imputation strategies, in an experimental setup with 34 datasets from the medical context, injected with the MNAR mechanism (10% to 80% rates). The results were evaluated through the Mean Absolute Error, with the new method being the overall best in 71% of the datasets, significantly outperforming the remaining ones, particularly for high missing rates. Finally, a case study of a classification task with heart failure data was also conducted, where this method induced improvements in 50% of the classifiers.
- Contact: rdpereira@dei.uc.pt

### Notes
- The PMIVAE package follows the scikit-learn architecture, implementing the `fit()`, `transform()` and `fit_transform()` methods.
- The data to be imputed must be a NumPy Array.
- The categorical features must be binarized before running PMIVAE (e.g., through one-hot encoding).
- The missing data must be pre-imputed. The PMIVAE package performs pre-imputation using the features’ mean and mode for the continuous and categorical types, respectively. Alternatively, the data can be pre-imputed with other approach before being passed to the package.
- The Variational Autoencoder architecture can be customized through the `ConfigVAE` data class. 
- A detailed usage example is available in `tests/test_breast_cancer.py`.

### Quick Start Example
```python
import numpy as np
from pmivae import ConfigVAE, PMIVAE

data = np.asarray([[0.31, 0.22, np.nan, 0.78], [0.43, np.nan, 0.67, 0.98]])

vae_config = ConfigVAE()
vae_config.input_shape = (4, )

pmivae_model = PMIVAE(vae_config, num_samples=200)
data_imputed = pmivae_model.fit_transform(data)
```