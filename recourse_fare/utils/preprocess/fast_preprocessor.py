from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import pandas as pd
import numpy as np

class StandardPreprocessor():

    def __init__(self,  exclude: list=[]) -> None:
        self.exclude = exclude
        self.feature_names_ordering = None
        self.numeric_columns = []
        self.categorical_columns = []
        self.onehot = OneHotEncoder(handle_unknown="ignore")
        self.scaler = MinMaxScaler()
    
    def fit(self, data: pd.DataFrame):

        self.feature_names_ordering = list(set(data.columns)-set(self.exclude))

        # Get categorical and numerical columns
        for c in self.feature_names_ordering:
            if is_numeric_dtype(data[c]):
                self.numeric_columns.append(c)
            elif is_string_dtype(data[c]):
                 self.categorical_columns.append(c)
            else:
                print(f"Skipping {c}. It is not string nor numeric.")
        
        self.onehot.fit(data[self.categorical_columns])
        self.scaler.fit(data[self.numeric_columns])

        self.feature_names_ordering = self.numeric_columns.copy()
        self.feature_names_ordering += self.onehot.get_feature_names_out(input_features=self.categorical_columns).tolist()

    def transform(self, data: pd.DataFrame, type: str="values"):

        transformed = data.copy()

        cat_ohe = self.onehot.transform(transformed[self.categorical_columns]).toarray()
        transformed[self.numeric_columns] = self.scaler.transform(transformed[self.numeric_columns])

        ohe_df = pd.DataFrame(cat_ohe, columns=self.onehot.get_feature_names_out(input_features=self.categorical_columns))
        transformed = pd.concat([transformed[self.numeric_columns], ohe_df], axis=1)
        
        if type == "values":
            return transformed.values
        else:
            return transformed
    
    def transform_dict(self, data: dict, type="values") -> dict:
        transformed = pd.DataFrame.from_records([data.copy()])
        return self.transform(transformed, type)

class FastPreprocessor:

    def __init__(self, exclude: list=[]) -> None: 
        """Class constructor.

        :param exclude: columns name we want to exclude from preprocessing, defaults to []
        :type exclude: list, optional
        """

        self.constants = {}
        self.inverse_constants = {}
        self.feature_names_ordering = None

        self.numerical_cols = set()
        self.categorical_cols = set()

        self.exclude=exclude

    def fit(self, data: pd.DataFrame):

        self.feature_names_ordering = list(set(data.columns)-set(self.exclude))

        for c in self.feature_names_ordering:
            if is_numeric_dtype(data[c]):
                self.numerical_cols.add(c)
                self.constants[c] = [
                    data[c].min(), data[c].max()
                ]
            elif is_string_dtype(data[c]):
                 self.categorical_cols.add(c)
                 self.constants[c] = { v:k for k,v, in enumerate(data[c].unique())}
                 self.inverse_constants[c] = { k:v for k,v, in enumerate(data[c].unique())}
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
                transformed[c] = transformed[c] if transformed[c] >= 1 else 1
                transformed[c] = transformed[c] if transformed[c] <= 2 else 2
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
    
    def inverse_transform(self, data: pd.DataFrame, type="values") -> pd.DataFrame:
        transformed = data.copy()

        def return_correct_key(k, keys):
            min_val, max_val = max(keys), min(keys)
            if k < min_val:
                return 1
            elif k > max_val:
                return max_val
            else:
                return k

        for c in self.feature_names_ordering:
            if c in self.inverse_constants:
                transformed[c] = transformed[c].apply(
                    lambda x: self.inverse_constants[c].get(return_correct_key(int(x)-1, list(self.inverse_constants[c].keys())))
                )
            elif c in self.constants:
                min_val, max_val = self.constants.get(c)
                transformed[c] = (transformed[c]-1)*(max_val-min_val)+min_val
                transformed[c] = transformed[c].apply(lambda x: min_val if x < min_val else x)
                transformed[c] = transformed[c].apply(lambda x: max_val if x > max_val else x)
        
        if type == "values":
            return transformed.values
        else:
            return transformed