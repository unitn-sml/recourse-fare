from ...utils.wfare_utils.structural_weights import StructuralWeights

import pandas as pd

class AdultSCM(StructuralWeights):

    def __init__(self, preprocessor, random_type=None):
        self.preprocessor = preprocessor
        self.random_type = random_type if random_type else -1
        super().__init__()

    def set_scm_structure(self, scm_structure=-1):
        self.random_type = scm_structure
        super().__init__()

    def _init_structure(self) -> None:
        self.scm.add_nodes_from(
            ["workclass",
            "education",
            "fnlwgt",
            "marital_status",
            "occupation",
            "education_num",
            "capital_gain",
            "capital_loss",
            "hours_per_week"]
        )

        if self.random_type == 1:
            self.scm.add_edges_from(
                [
                ("education", "workclass"),
                ("hours_per_week", "occupation"),
                ("workclass", "capital_gain"),
                ("capital_gain", "capital_loss")
                ]
            )
        elif self.random_type == 2:
            self.scm.add_edges_from(
                [
                ("workclass", "occupation"),
                ("workclass", "hours_per_week"),
                ("workclass", "occupation"),
                ]
            )
        elif self.random_type == 3:
            self.scm.add_edges_from(
                [
                ("education", "workclass"),
                ("workclass", "occupation"),
                ("workclass", "occupation"),
                ("capital_gain", "capital_loss")
                ]
            )
        else:
            self.scm.add_edges_from(
                [
                ("education", "workclass"),
                ("workclass", "occupation"),
                ("hours_per_week", "occupation"),
                ("workclass", "hours_per_week"),
                ("workclass", "capital_gain"),
                ("workclass", "occupation"),
                ("capital_gain", "capital_loss")
                ]
            )
    
    def _feature_mapping(self, features: dict) -> dict:
        return self.preprocessor.transform_dict(features, type="raw")

