from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import pandas as pd
import numpy as np

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