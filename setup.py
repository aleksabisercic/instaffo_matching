from setuptools import setup, find_packages

setup(
    name='instaffo_matching',
    version='0.1.0',
    description='A project for data loading, preprocessing, feature engineering, and matching talent and jobs.',
    author='Aleksa Bisercic',
    author_email='aleksabisercic@gmail.com',
    packages=find_packages(), 
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        # Add other dependencies as needed
    ],
    entry_points={
        'console_scripts': [
            # Command-line scripts
            # 'instaffo-train=scripts.train_model:main',
            # 'instaffo-evaluate=scripts.evaluate_model:main',
        ],
    },
)