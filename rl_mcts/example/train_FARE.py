"""Train the FARE model by using the given python class instead of the script."""

from rl_mcts.core.models.FARE import FARE

import pandas as pd

if __name__ == "__main__":

    config_path = "rl_mcts/example/config_base_scm.yml"

    X_test = pd.read_csv("rl_mcts/example/data.csv")[0:10]
    Y = X_test["sum_total"]
    X_test.drop(columns="sum_total", inplace=True)

    X_weights = pd.DataFrame()

    model = FARE(config_path, model_path="fare.pth")
    model.fit(max_iter=2, verbose=False)
    model.save("fare.pth")

    Y = model.predict([X_test, X_weights])
    print(Y)