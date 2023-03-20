# Recourse-FARE: (Explainable) Algorithmic Recourse with Reinforcement Learning and MCTS

This library provides a set of methods which can be used to achieve model-agnostic algorithmic recourse given a black-box model. The library enables customization of all the aspects of the recourse process, from the actions available to the models employed.

**If you want to have a gist of a practical application and how to use `recourse-fare`, please have a look at the following notebook [Tutorial: training FARE and E-FARE models on the Adult dataset](./recourse_fare/example/notebooks/train_fare_adult.ipynb)**

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

## References

We use the library in the following projects: 

[1] De Toni, Giovanni, Bruno Lepri, and Andrea Passerini. "Synthesizing explainable counterfactual policies for algorithmic recourse with program synthesis." Machine Learning (2023): 1-21, [10.1007/s10994-022-06293-7](https://link.springer.com/article/10.1007/s10994-022-06293-7)

[2] De Toni, Giovanni, et al. "User-Aware Algorithmic Recourse with Preference Elicitation." arXiv preprint arXiv:2205.13743 (2022), [2205.13743](https://arxiv.org/abs/2205.13743)