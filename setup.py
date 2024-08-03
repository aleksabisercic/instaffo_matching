from setuptools import find_packages, setup

setup(
    name="instaffo_matching",
    version="0.1.0",
    description="A project for data loading, preprocessing, feature engineering, and matching talent and jobs.",
    author="Aleksa Bisercic",
    author_email="aleksabisercic@gmail.com",
    packages=find_packages(),
    install_requires=["numpy", "pandas>=2.0.0", "scikit-learn>=1.0", "joblib==1.4.2"],
    extras_require={
        "dev": [
            "matplotlib>=3.4.3",
            "seaborn>=0.11.2",
            "jupyterlab>=3.1.12",
            "ipykernel>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            # Command-line scripts
            # 'instaffo-train=scripts.train_model:main',
            # 'instaffo-evaluate=scripts.evaluate_model:main',
        ],
    },
)
