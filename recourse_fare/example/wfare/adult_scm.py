from ...utils.wfare_utils.structural_weights import StructuralWeights

import pandas as pd

class AdultSCM(StructuralWeights):

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
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

