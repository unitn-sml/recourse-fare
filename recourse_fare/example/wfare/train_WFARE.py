"""Train the FARE model by using the given python class instead of the script."""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from recourse_fare.models.WFARE import WFARE

from recourse_fare.example.wfare.adult_scm import AdultSCM

import pandas as pd
import numpy as np

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import os

class FastPreprocessor:

    def __init__(self, numerical_encoding="minmax", categorical_encoding="ordinal") -> None:        
        self.constants = {}
        self.feature_names_ordering = None

        self.numerical_cols = set()
        self.categorical_cols = set()

        self.num_encoding = numerical_encoding
        self.cat_encoding = categorical_encoding

    def fit(self, data: pd.DataFrame):

        self.feature_names_ordering = list(data.columns)

        for c in self.feature_names_ordering:
            if is_numeric_dtype(data[c]):
                self.numerical_cols.add(c)
                self.constants[c] = [
                    data[c].min(), data[c].max()
                ]
            elif is_string_dtype(data[c]):
                 self.categorical_cols.add(c)
                 self.constants[c] = { v:k for k,v, in enumerate(data[c].unique())}
            else:
                print(f"Skipping {c}. It is not string nor numeric.")
    
    def get_feature_names_out(self):
        return self.feature_names_ordering

    def transform_dict(self, data: dict, type="values") -> dict:
        transformed = data.copy()

        for c in self.feature_names_ordering:
            if c in self.numerical_cols:
                min_val, max_val = self.constants.get(c)
                transformed[c] = 1+(transformed[c]-min_val)/(max_val-min_val)
            elif c in self.categorical_cols:
                transformed[c] = 1+self.constants[c].get(transformed[c], 0)
        
        if type == "values":
            return np.array([
               transformed[c] for c in self.feature_names_ordering
            ])
        else:
            return transformed

    def transform(self, data: pd.DataFrame, type="values") -> pd.DataFrame:
        transformed = data.copy()

        for c in self.feature_names_ordering:
            if is_numeric_dtype(data[c]):
                min_val, max_val = self.constants.get(c)
                transformed[c] = 1+(transformed[c]-min_val)/(max_val-min_val)
            elif is_string_dtype(data[c]):
                transformed[c] = transformed[c].apply(
                    lambda x: 1+self.constants[c].get(x, 0)
                )
        
        if type == "values":
            return transformed.values
        else:
            return transformed

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
    X.drop(columns=["fnlwgt", "age", "sex", "native_country", "relationship", "education_num"], inplace=True)

    # Split the dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Generate random weights. Weights needs to be non-null and positive
    single_weights = np.ones(15)
    W_train = [single_weights for _ in range(len(X_train))]
    W_test = np.abs(np.random.normal(loc=0, size=(len(X_test), 15)))+1

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

    # Filter the training dataset by picking only the examples which are classified negatively by the model
    output = blackbox_model.predict(preprocessor.transform(X_train))
    X_train["predicted"] = output
    X_train = X_train[X_train.predicted == 1]
    X_train.drop(columns="predicted", inplace=True)

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

    # Train a FARE model given the previous configurations
    model = WFARE(blackbox_model, policy_config, environment_config, mcts_config, batch_size=50)
    if not os.path.isfile("fare.pth"):
        model.fit(X_train, W_train, max_iter=1000, tensorboard="./wfare")
        # We save the trained FARE model to disc
        model.save("fare.pth")
    else:
        model.load("fare.pth")

    # For testing, we use the test data
    output = blackbox_model.predict(preprocessor.transform(X_test))
    X_test["predicted"] = output
    X_test = X_test[X_test.predicted == 1]
    X_test.drop(columns="predicted", inplace=True)

    # We use the model to predict the test data
    _, Y_full, _, _, _ = model.predict(X_test[0:100], W_test[0:100], full_output=True)
    _, Y_agent, _, _, _ = model.predict(X_test[0:100], W_test[0:100], full_output=True, agent_only=True)
    _, Y_mcts, _, _, _ = model.predict(X_test[0:100], W_test[0:100], full_output=True, mcts_only=True)
    
    print(accuracy_score(Y_full, y_test[:100]))
    print(accuracy_score(Y_agent, y_test[:100]))
    print(accuracy_score(Y_mcts, y_test[:100]))