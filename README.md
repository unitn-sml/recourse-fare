# RL-MCTS: Reinforcement Learning with MCTS

This is a simple library containing the code for training an RL agent by using Monte Carlo Tree Search. 

## Usage

First, we need to clone this repository locally. We also need to create a suitable conda environment with all the dependencies needed. We provide an `environment.yml` file with the corresponding packages and files. 

```bash
git clone https://github.com/unitn-sml/rl-mcts.git
cd rl-mcts
conda env create -f environment.yml
conda activate recourse_fare
pip install .
```

Then, we can either install the library in our system or we can include it as a "third-party" directory in our project and user it directly from there. 

We use the library in the following projects: 

[1] De Toni, Giovanni, Bruno Lepri, and Andrea Passerini. "Synthesizing explainable counterfactual policies for algorithmic recourse with program synthesis." Machine Learning (2023): 1-21.

[2] De Toni, Giovanni, et al. "User-Aware Algorithmic Recourse with Preference Elicitation." arXiv preprint arXiv:2205.13743 (2022).