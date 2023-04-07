"""Train the FARE model by using the given python class instead of the script."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from recourse_fare.models.WFARE import WFARE

from recourse_fare.example.wfare.adult_scm import AdultSCM
from recourse_fare.utils.preprocess.fast_preprocessor import FastPreprocessor
from recourse_fare.utils.Mixture import MixtureModel

import pandas as pd
import numpy as np

import os

import dill as pickle

if __name__ == "__main__":

    # Seed for reproducibility
    np.random.seed(2023)

    # Read data and preprocess them
    X = pd.read_csv("recourse_fare/example/wfare/data.csv")
    y = X.income_target.apply(lambda x: 1 if x=="<=50K" else 0)
    X.drop(columns=["income_target", "predicted"], inplace=True)

    # We drop some columns we do not consider actionable. It makes the problem less interesting, but it does
    # show the point about how counterfactual interventions works. 
    #X.drop(columns=["fnlwgt", "age", "race", "sex", "native_country", "relationship", "education_num"], inplace=True)
    #X.drop(columns=["fnlwgt", "age", "sex", "race", "native_country", "relationship", "education_num"], inplace=True)
    X.drop(columns=["fnlwgt", "age", "sex", "race", "native_country", "relationship", "education_num"], inplace=True)

    # Split the dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Save the training/testing datasets
    X_train.to_csv("train_data.csv", index=None)
    X_test.to_csv("test_data.csv", index=None)

    # Generate random weights. Weights needs to be non-null and positive
    mixture = MixtureModel(dimensions=15)
    single_weights = mixture.sample(1)
    W_train = [single_weights for _ in range(len(X_train))]
    W_test = mixture.sample(len(X_test))

    # Build weights dataframes
    tmp_scm = AdultSCM(None)
    keys_weights = [(node, node) for node in tmp_scm.scm.nodes()]
    keys_weights += [(parent, node) for parent,node in tmp_scm.scm.edges()]

    W_train = pd.DataFrame(W_train, columns=keys_weights)
    W_test = pd.DataFrame(W_test, columns=keys_weights)

    # Save weights to disk
    W_test.to_csv("weights_test.csv", index=None)

    # Build a preprocessing pipeline, which can be used to preprocess
    # the elements of the dataset.
    # The Fast preprocessor does min/max scaling and categorical encoding.
    # It is much faster than then scikit learn ones and it uses dictionaries
    # and sets to perform operations on the fly.
    preprocessor = FastPreprocessor()
    preprocessor.fit(X_train)

    # Fit a simple SVC model over the data
    blackbox_model = SVC(class_weight="balanced")
    blackbox_model.fit(preprocessor.transform(X_train), y_train)

    # Evaluate the model and print the classification report for the two classes
    output = blackbox_model.predict(preprocessor.transform(X_test))
    print(classification_report(output, y_test))

    # Filter the training dataset by picking only the examples which are classified negatively by the model
    output = blackbox_model.predict(preprocessor.transform(X_train))
    X_train["predicted"] = output
    X_train = X_train[X_train.predicted == 1]
    X_train.drop(columns="predicted", inplace=True)

    policy_config= {
        "observation_dim": 7,
        "encoding_dim": 15,
        "hidden_size": 15
    }

    environment_config = {
        "class_name": "recourse_fare.example.wfare.mock_adult_env.AdultEnvironment",
        "additional_parameters": {
            "preprocessor": preprocessor
        }
    }
    
    mcts_config = {
        "exploration": True,
        "number_of_simulations": 15,
        "dir_epsilon": 0.3,
        "dir_noise": 0.3
    }

    # Train a FARE model given the previous configurations
    model = WFARE(blackbox_model, policy_config, environment_config, mcts_config, batch_size=100)
    if not os.path.isfile("fare.pth"):
        model.fit(X_train, W_train, max_iter=1000, tensorboard="./wfare")
        # We save the trained FARE model to disc
        model.save("fare.pth")
    else:
        model.load("fare.pth")
    pickle.dump(model, open("recourse.pth", "wb"))

    # For testing, we use the test data
    output = blackbox_model.predict(preprocessor.transform(X_test))
    X_test["predicted"] = output
    X_test = X_test[X_test.predicted == 1]
    X_test.drop(columns="predicted", inplace=True)

    # We use the model to predict the test data
    _, Y_full, _, _, _ = model.predict(X_test[0:100], W_test[0:100], full_output=True)
    _, Y_agent, _, _, _ = model.predict(X_test[0:100], W_test[0:100], full_output=True, agent_only=True)
    _, Y_mcts, _, _, _ = model.predict(X_test[0:100], W_test[0:100], full_output=True, mcts_only=True)
    
    print(sum(Y_full)/len(Y_full))
    print(sum(Y_agent)/len(Y_agent))
    print(sum(Y_mcts)/len(Y_mcts))