from setuptools import setup, find_packages

setup(
    name="autodml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "feature-engine",
        "cloudpickle",
        "fastapi",
        "uvicorn",
        "matplotlib",
        "seaborn",
        "optuna",
        "nltk",
        "",
    ],
    include_package_data=True,
)
