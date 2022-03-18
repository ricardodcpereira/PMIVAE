from setuptools import setup

setup(
    name='PMIVAE',
    version='1.0',
    packages=['pmivae'],
    url='https://ricardodcpereira.com',
    license='MIT',
    author='Ricardo Pereira',
    author_email='rdpereira@dei.uc.pt',
    description='PMIVAE - Partial Multiple Imputation with Variational Autoencoders',
    python_requires='>=3.6.*',
    install_requires=['numpy>=1.19.5', 'scikit-learn>=0.24.2'],
    extras_require={
        'tf': ['tensorflow>=2.5.0'],
        'tf_gpu': ['tensorflow-gpu>=2.5.0'],
    }
)
