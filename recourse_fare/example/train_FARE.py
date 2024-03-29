"""Train the FARE model by using the given python class instead of the script."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from recourse_fare.models.FARE import FARE

import pandas as pd
import numpy as np

def model(features):
    return np.sum([features.get(k) for k in features.keys()])

if __name__ == "__main__":

    # Read data and preprocess them
    X = pd.read_csv("recourse_fare/example/data.csv")
    y = X["sum_total"]
    X.drop(columns="sum_total", inplace=True)

    # Split the dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    # Train a standard scaler over the data
    # We will give this to the FARE model to standardize the observations
    scaler = StandardScaler()
    scaler.fit(X_train)

    policy_config= {
        "observation_dim": 5,
        "encoding_dim": 20,
        "hidden_size": 50,
        "model_path": None
    }

    environment_config = {
        "class_name": "recourse_fare.example.mock_env_scm.MockEnv",
        "additional_parameters": {
            "preprocessing": scaler
        }
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


    # Train a FARE model given the previous configurations
    model = FARE(model, policy_config, environment_config, mcts_config)
    model.fit(X_train, max_iter=10)
    
    # We save the trained FARE model to disc
    model.save("fare.pth")

    # We use the model to predict the test data
    Y = model.predict(X_test[0:10])
    print(Y)