from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import pandas as pd
import numpy as np

class StandardPreprocessor():

    def __init__(self,  exclude: list=[]) -> None:

        self.exclude = exclude
        self.feature_names_ordering = None
        self.original_feature_names_ordering = None
        
        self.continuous = []
        self.categorical = []

        self.onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.scaler = MinMaxScaler()
    
    def get_feature_names(self, feature_names: list):
        return self.onehot.get_feature_names_out(input_features=feature_names).tolist()
    
    def fit(self, data: pd.DataFrame):

        self.feature_names_ordering = list(set(data.columns)-set(self.exclude))
        self.original_feature_names_ordering = self.feature_names_ordering.copy()

        # Get categorical and numerical columns
        for c in self.feature_names_ordering:
            if is_numeric_dtype(data[c]):
                self.continuous.append(c)
            elif is_string_dtype(data[c]):
                 self.categorical.append(c)
            else:
                print(f"Skipping {c}. It is not string nor numeric.")
        
        if self.continuous:
            self.scaler.fit(data[self.continuous].values)
            self.feature_names_ordering = self.continuous.copy()

        if self.categorical:
            self.onehot.fit(data[self.categorical].values)
            self.feature_names_ordering += self.onehot.get_feature_names_out(input_features=self.categorical).tolist()

    def transform(self, data: pd.DataFrame, type: str="values"):

        transformed = data.copy()
        transformed.reset_index(drop=True, inplace=True)

        transformed[self.continuous] = self.scaler.transform(transformed[self.continuous].values)

        if self.categorical:
            cat_ohe = self.onehot.transform(transformed[self.categorical].values)

            ohe_df = pd.DataFrame(cat_ohe, columns=self.onehot.get_feature_names_out(input_features=self.categorical))
            transformed = pd.concat([transformed[self.continuous], ohe_df], axis=1)

        if type == "values":
            return transformed.values
        else:
            return transformed[self.feature_names_ordering]
    
    def inverse_transform(self, data: pd.DataFrame, type: str="values"):

        transformed = data.copy()
        transformed.reset_index(drop=True, inplace=True)

        cat_trans_name = self.onehot.get_feature_names_out(input_features=self.categorical)

        cat_ohe = self.onehot.inverse_transform(transformed[cat_trans_name].values)
        ohe_df = pd.DataFrame(cat_ohe, columns=self.categorical)
        transformed[self.continuous] = self.scaler.inverse_transform(transformed[self.continuous].values)
        transformed = pd.concat([transformed[self.continuous], ohe_df], axis=1)

        if type == "values":
            return transformed.values
        else:
            return transformed
    
    def transform_dict(self, data: dict, type="values") -> dict:

        categorical = np.array([data.get(k) for k in self.categorical])
        continuous = np.array([data.get(k) for k in self.continuous])

        cat_ohe = self.onehot.transform([categorical])[0]
        cont = self.scaler.transform([continuous])[0]

        transformed = {
            k:v for k,v in zip(self.onehot.get_feature_names_out(input_features=self.categorical), cat_ohe)
        }
        for k,v in zip(self.continuous, cont):
            transformed[k] = v
        
        if type == "values":
            return np.array([
               transformed[c] for c in self.feature_names_ordering
            ])
        else:
            return transformed

class FastPreprocessor:

    def __init__(self, exclude: list=[]) -> None: 
        """Class constructor.

        :param exclude: columns name we want to exclude from preprocessing, defaults to []
        :type exclude: list, optional
        """

        self.categorical_encoded = []
        self.constants = {}
        self.inverse_constants = {}

        self.original_feature_name_ordering = []
        self.feature_names_ordering = []

        self.continuous = []
        self.categorical = []

        self.exclude=exclude

    def fit(self, data: pd.DataFrame):

        self.original_feature_name_ordering = data.columns

        for c in self.original_feature_name_ordering:
            if is_numeric_dtype(data[c]):
                self.feature_names_ordering.append(c)
                self.continuous.append(c)
                self.constants[c] = [
                    data[c].min(), data[c].max()
                ]
            elif is_string_dtype(data[c]):
                
                categorical_unique_values = sorted(data[c].unique())

                self.categorical.append(c)
                self.categorical_encoded += [f"{c}_{v}" for v in categorical_unique_values]
                self.feature_names_ordering += [f"{c}_{v}" for v in categorical_unique_values]
                
                self.constants[c] = { v:k for k,v in enumerate(categorical_unique_values)}
                self.inverse_constants[c] = { k:v for k,v in enumerate(categorical_unique_values)}
            else:
                print(f"Skipping {c}. It is not string nor numeric.")
    
    def get_feature_names_out(self, feature_names=None):
        return self.feature_names_ordering

    def transform_dict(self, data: dict, type="values") -> dict:
        
        transformed = []

        for c in self.original_feature_name_ordering:
            if c in self.continuous:
                min_val, max_val = self.constants.get(c)
                encoded_value = (data[c]-min_val)/(max_val-min_val)
                transformed.append(min(1, max(0, encoded_value)))
            elif c in self.categorical:
                encoded_value = np.zeros(len(self.constants[c])).tolist()
                idx_to_set = self.constants[c].get(data[c], -1)
                if idx_to_set != -1:
                    encoded_value[idx_to_set] = 1                

                transformed += encoded_value

        if type == "values":
            return np.array(transformed)
        else:
            return {k:v for k,v in zip(self.feature_names_ordering, transformed)}

    def transform(self, data: pd.DataFrame, type="values") -> pd.DataFrame:
        
        transformed = data.copy()
        transformed = transformed.to_dict('records')

        encoded_data = []
        for record in transformed:
            encoded_data.append(
                self.transform_dict(record, type)
            )

        if type == "values":
            return np.array(encoded_data)
        else:
            return pd.DataFrame.from_records(encoded_data)
    
    def get_one_hot_index(self, data: dict, feature:str):
        for k,v in self.constants[feature].items():
            if data.get(f"{feature}_{k}") == 1:
                return v
        return -1
    
    def inverse_transform_dict(self, data: dict, type="values"):

        transformed = {}
        
        for c in self.feature_names_ordering:
            v = data.get(c)
            if c in self.continuous:
                min_val, max_val = self.constants.get(c)
                transformed[c] = v*(max_val-min_val)+min_val
                assert min_val <= transformed[c] <= max_val
            elif c in self.categorical_encoded:
                for c_original in self.categorical:
                    if v == 1:
                        if c_original in c:
                            value = c.replace(f"{c_original}_", "")    
                            transformed[c_original] = value
                            break

        if type == "values":
            return [transformed.get(c, None) for c in self.original_feature_name_ordering]
        else:
            return {v:transformed.get(v, None) for v in self.original_feature_name_ordering}

    def inverse_transform(self, data: pd.DataFrame, type="values") -> pd.DataFrame:
        
        transformed = data.copy()
        transformed = transformed.to_dict('records')

        encoded_data = []
        for record in transformed:
            encoded_data.append(
                self.inverse_transform_dict(record, type)
            )

        if type == "values":
            return np.array(encoded_data)
        else:
            return pd.DataFrame.from_records(encoded_data)