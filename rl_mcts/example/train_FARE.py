"""Train the FARE model by using the given python class instead of the script."""

from sklearn.model_selection import train_test_split

from rl_mcts.core.models.FARE import FARE

import pandas as pd

if __name__ == "__main__":

    config_path = "rl_mcts/example/config_base_scm.yml"

    X = pd.read_csv("rl_mcts/example/data.csv")
    y = X["sum_total"]
    X.drop(columns="sum_total", inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    model = FARE(config_path)
    model.fit(X_train, y_train, max_iter=2, verbose=False)
    model.save("fare.pth")

    Y = model.predict(X_test)
    print(Y)