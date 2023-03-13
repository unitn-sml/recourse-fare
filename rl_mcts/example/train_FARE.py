"""Train the FARE model by using the given python class instead of the script."""

from sklearn.model_selection import train_test_split

from rl_mcts.core.models.FARE import FARE

import pandas as pd

if __name__ == "__main__":

    policy_config= {
        "observation_dim": 5,
        "encoding_dim": 20,
        "hidden_size": 50,
        "model_path": None
    }

    environment_config = {
        "class_name": "rl_mcts.example.mock_env_scm.MockEnv",
        "additional_parameters": {}
    }
    
    mcts_config = {
        "exploration": True,
        "number_of_simulations": 10,
        "dir_epsilon": 0.03,
        "dir_noise": 0.3,
        "level_closeness_coeff": 3.0,
        "level_0_penalty": 1.0,
        "qvalue_temperature": 1.0,
        "temperature": 1.3,
        "c_puct": 0.5,
        "gamma": 0.97
    }

    # Read data and preprocess them
    X = pd.read_csv("rl_mcts/example/data.csv")
    y = X["sum_total"]
    X.drop(columns="sum_total", inplace=True)

    # Split the dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    # Train a FARE model given the previous configurations
    model = FARE(policy_config, environment_config, mcts_config)
    model.fit(X_train, y_train, max_iter=2, verbose=False)
    
    # We save the trained FARE model to disc
    model.save("fare.pth")

    # We use the model to predict the test data
    Y = model.predict(X_test[0:10])
    print(Y)