# Recourse-FARE: (Explainable) Algorithmic Recourse with Reinforcement Learning and MCTS

This library provides a set of methods which can be used to achieve model-agnostic algorithmic recourse given a black-box model. The library enables customization of all the aspects of the recourse process, from the actions available to the models employed.

**If you want to have a gist of a practical application and how to use `recourse-fare`, please have a look at the following notebook [Tutorial: training FARE and E-FARE models on the Adult dataset](https://github.com/unitn-sml/recourse-fare/blob/master/recourse_fare/example/notebooks/train_fare_adult.ipynb)**

## Install

The library can be easily installed from GitHub directly. We suggest using a virtualenv to make it easier to develop on top of it.

```bash
pip install git+https://github.com/unitn-sml/recourse-fare.git@v0.1.0
```

## Development

We can easily download the following library and install it locally in your preferred (virtual) environment. We suggest using **Python 3.7** and **conda**. If you find any issue with the following procedure, feel free to open an issue!

```bash
git clone https://github.com/unitn-sml/recourse-fare.git
cd recourse-fare
conda create --name recourse_fare python=3.7
conda activate recourse_fare
pip install -e .
```

## How to cite

If you are using our library, please consider citing the following works:

```
@article{
    detoni2024personalized,
    title={Personalized Algorithmic Recourse with Preference Elicitation},
    author={Giovanni De Toni and Paolo Viappiani and Stefano Teso and Bruno Lepri and Andrea Passerini},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2024},
    url={https://openreview.net/forum?id=8sg2I9zXgO},
    note={}
}

@article{detoni2023synthesizing,
  title={Synthesizing explainable counterfactual policies for algorithmic recourse with program synthesis},
  author={De Toni, Giovanni and Lepri, Bruno and Passerini, Andrea},
  journal={Machine Learning},
  volume={112},
  number={4},
  pages={1389--1409},
  year={2023},
  publisher={Springer}
}
```