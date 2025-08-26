from setuptools import setup, find_packages

setup(
    name="job-recommendation-system",
    version="1.0.0",
    description="A machine learning-based job recommendation system for matching students with job opportunities",
    author="Rishabh Sahni",
    author_email="f20211630@pilani.bits-pilani.ac.in",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.2.0",
        "nltk>=3.8",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "joblib>=1.3.0",
        "fuzzywuzzy>=0.18.0",
        "mlflow>=2.7.0",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)