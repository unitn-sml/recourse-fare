"""Train the FARE model by using the given python class instead of the script."""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from recourse_fare.models.WFARE import WFARE
from recourse_fare.models.InteractiveFARE import InteractiveFARE

from recourse_fare.example.wfare.adult_scm import AdultSCM
from recourse_fare.user.user import NoiselessUser
from recourse_fare.utils.preprocess.fast_preprocessor import FastPreprocessor

import pandas as pd
import numpy as np

import pickle

if __name__ == "__main__":

    # Seed for reproducibility
    np.random.seed(2023)

    # Read data and preprocess them
    X = pd.read_csv("recourse_fare/example/wfare/data.csv")[0:5000]
    y = X.income_target.apply(lambda x: 1 if x=="<=50K" else 0)
    X.drop(columns=["income_target", "predicted"], inplace=True)

    # We drop some columns we do not consider actionable. It makes the problem less interesting, but it does
    # show the point about how counterfactual interventions works. 
    #X.drop(columns=["fnlwgt", "age", "race", "sex", "native_country", "relationship", "education_num"], inplace=True)
    X.drop(columns=["fnlwgt", "age", "sex", "native_country", "relationship", "education_num"], inplace=True)

    # Split the dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Generate random weights. Weights needs to be non-null and positive
    single_weights = np.ones(15)
    W_train = [single_weights for _ in range(len(X_train))]
    W_test = np.abs(np.random.normal(loc=25, size=(len(X_test), 15)))

    # Build weights dataframes
    tmp_scm = AdultSCM(None)
    keys_weights = [(node, node) for node in tmp_scm.scm.nodes()]
    keys_weights += [(parent, node) for parent,node in tmp_scm.scm.edges()]

    W_train = pd.DataFrame(W_train, columns=keys_weights)
    W_test = pd.DataFrame(W_train, columns=keys_weights)

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

    policy_config= {
        "observation_dim": 8,
        "encoding_dim": 25,
        "hidden_size": 25
    }

    environment_config = {
        "class_name": "recourse_fare.example.wfare.mock_adult_env.AdultEnvironment",
        "additional_parameters": {
            "preprocessor": preprocessor
        }
    }
    
    mcts_config = {
        "exploration": True,
        "number_of_simulations": 10,
        "dir_epsilon": 0.3,
        "dir_noise": 0.3
    }

    # For testing, we use the test data
    output = blackbox_model.predict(preprocessor.transform(X_test))
    X_test["predicted"] = output
    X_test = X_test[X_test.predicted == 1]
    X_test.drop(columns="predicted", inplace=True)

    # Filter the weights to keep the one we care
    W_test = W_test.iloc[X_test.index] 

    # Load a FARE model given the previous configurations
    model = WFARE(blackbox_model, policy_config, environment_config, mcts_config, batch_size=50)
    model.load("fare.pth")

    with open("recourse.pth", "wb") as f:
        pickle.dump(model, f)

    # Create the user model required
    user = NoiselessUser()

    # Create and interactive FARE object and predict the test instances
    interactive = InteractiveFARE(model, user, keys_weights, questions=5, verbose=True)
    (counterfactuals, Y, traces, costs, _), W_updated, failed_users = interactive.predict(X_test[0:2], W_test[0:2], full_output=True)

    print(counterfactuals)
    print(Y)
    print(costs)
    print(failed_users)
    print(W_updated.to_dict("records")[1])
    print(W_test.to_dict("records")[1])
