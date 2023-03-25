import pathlib
from setuptools import find_packages, setup

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name="recourse-fare",
    version="0.1.0",
    description="lgorithmic Recourse with Reinforcement Learning and MCTS (Structured Machine Learning Lab)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/unitn-sml/recourse-fare",
    author="Giovanni De Toni",
    author_email="giovanni.detoni@unitn.it",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=0.23.4",
        "numpy>=1.15.4",
        "colorama>=0.4.6",
        "scikit-learn>=0.23.2",
        "torch>=1.7.0",
        "causalgraphicalmodels==0.0.4",
        "tensorboardX==2.6",
        "tqdm==4.64.1",
        "dill==0.3.6"
    ]
)
