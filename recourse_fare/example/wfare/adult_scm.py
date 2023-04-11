from ...utils.wfare_utils.structural_weights import StructuralWeights

import pandas as pd

DEFAULT_NODES = ["workclass",
            "education",
            "fnlwgt",
            "marital_status",
            "occupation",
            "education_num",
            "capital_gain",
            "capital_loss",
            "hours_per_week"]

DEFAULT_EDGES = [
                ("education", "workclass"),
                ("workclass", "occupation"),
                ("hours_per_week", "occupation"),
                ("workclass", "hours_per_week"),
                ("workclass", "capital_gain"),
                ("workclass", "occupation"),
                ("capital_gain", "capital_loss")
                ]

class AdultSCM(StructuralWeights):

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        super().__init__()

    def _init_structure(self, nodes: list=None, edges: list=None) -> None:

        self.scm.add_nodes_from(
            nodes if nodes else DEFAULT_NODES
        )
            
        self.scm.add_edges_from(
            edges if edges else DEFAULT_EDGES
        )
        
    
    def _feature_mapping(self, features: dict) -> dict:
        return self.preprocessor.transform_dict(features, type="raw")

