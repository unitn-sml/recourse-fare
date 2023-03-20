import pandas as pd
import numpy as np

class DataLoader():
    
    def __init__(self, dataset=None,
                 weights_dataset=None,
                 X=None, y=None,
                 bad_class_value="bad", target_column="loan", predicted_column="loan",
                 deterministic=False):

        if X is not None and y is not None:
            self.data = X
            self.y = y

        else:

            user_features_dataset = pd.read_csv(dataset)

            # Filter the elements to pick only the ones with bad classification
            filter_only_bad_elements = (user_features_dataset[target_column] == bad_class_value) & (
                        user_features_dataset[predicted_column] == bad_class_value)
            user_features_dataset = user_features_dataset[filter_only_bad_elements]

            self.y = user_features_dataset[predicted_column]
            self.y.reset_index(drop=True, inplace=True)
            user_features_dataset.drop(columns=[predicted_column, target_column], inplace=True)

            user_features_dataset.reset_index(inplace=True, drop=True)

            self.data = user_features_dataset

        if weights_dataset is None:
            user_weight_dataset = None
        else:
            user_weight_dataset = pd.read_csv(weights_dataset)
            user_weight_dataset = user_weight_dataset.filter(items=user_features_dataset.index, axis=0)
            user_weight_dataset.reset_index(inplace=True, drop=True)

        self.weight_data = user_weight_dataset

        self.current_idx = 0

        self.failed_examples = []

        self.deterministic = deterministic

    def get_example(self, specific_idx: int=None, sample_errors=-1):

        # If deterministic, get the elements sequentially
        if specific_idx is not None:
            self.current_idx = specific_idx
        elif (np.random.rand(1)[0] > sample_errors or len(self.failed_examples) == 0) and not self.deterministic:
            self.current_idx = np.random.randint(0, len(self.data), 1)[0]
        elif not self.deterministic:
            self.current_idx = np.random.randint(0, len(self.failed_examples), 1)[0]
            features, weights = self.failed_examples[self.current_idx]
            self.failed_examples.pop(self.current_idx)

        features = self.data.iloc[[self.current_idx]].to_dict('records')[0]

        if self.weight_data:
            weights = self.weight_data.iloc[[self.current_idx]].to_dict('records')[0]
        else:
            weights = {}

        # Increment if it is deterministic
        if self.deterministic:
            self.current_idx += 1

        return features.copy(), weights.copy()

    def add_failed_example(self, features, weights):
       if len(self.failed_examples) > 500:
           self.failed_examples.pop()

       self.failed_examples.append((features, weights))
