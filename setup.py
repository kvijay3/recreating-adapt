from setuptools import setup, find_packages

setup(
    name="adapt_reimpl",
    version="0.1.0",
    description="From-scratch reimplementation of ADAPT and BADGERS for Cas13a diagnostic guide design",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.3",
        "numpy>=1.16",
        "scipy",
        "scikit-learn",
        "biopython",
        "primer3-py>=0.6.1",
        "pandas",
        "pulp",
        "editdistance",
    ],
)
