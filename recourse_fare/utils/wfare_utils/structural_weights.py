from abc import abstractmethod, ABC
from typing import Tuple

import networkx as nx

import numpy as np

class StructuralWeights(ABC):

    def __init__(self):
        self.scm = nx.DiGraph()
        self._init_structure()

    @abstractmethod
    def _init_structure(self) -> None:
        """
        Initialize the node and edge weights. Moreover, initialize for each
        node its parents.
        """
        pass 

    @abstractmethod
    def _feature_mapping(self, features: dict) -> dict:
        """
        Map the features to a suitable representation for the structural weights.
        :param features:
        :return: mapped features
        """
        pass

    def compute_cost(self, node: str, new_value, features: dict, weights: dict) -> float:
        """
        Compute the cost of changing one feature given
        :param name:
        :param new_value:
        :param features:
        :return:
        """

        # Apply feature mapping to all the features and to the new value
        features_tmp = features.copy()
        features_tmp[node] = new_value
        features = self._feature_mapping(features)
        new_value = self._feature_mapping(features_tmp).get(node)

        # Compute the cost given the parents
        cost = 0
        parent_edges = self.scm.predecessors(node)
        for parent in parent_edges:
            cost += weights.get((parent, node))*features.get(parent)

        # Return the cost plus the variation of the current value
        # If the clip the cost to be positive. A negative cost does not make sense.
        #assert (new_value-features.get(name))*self.node_weights.get(name) >= 0, f"{new_value}, {features.get(name)}, the cost is negative {name}"
        return max(0, cost + np.abs(new_value - features.get(node)) * weights.get((node, node)))
