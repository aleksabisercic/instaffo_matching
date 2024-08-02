from setuptools import setup, find_packages

setup(
    name='instaffo_matching',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
    ],
)
