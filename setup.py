import pathlib
from setuptools import find_packages, setup

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name="recourse-fare",
    version=1.0,
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
    include_package_data=True
)