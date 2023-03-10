"""Train the FARE model by using the given python class instead of the script."""

from rl_mcts.core.models.FARE import FARE

import pandas as pd

if __name__ == "__main__":

    config_path = "rl_mcts/example/config_base_scm.yml"

    X_test = pd.read_csv("rl_mcts/example/data.csv")

    model = FARE(config_path)
    model.fit(max_iter=10, verbose=False)

    model.predict(X_test)