from abc import abstractmethod, ABC
from typing import Tuple

import networkx as nx

import numpy as np

class StructuralWeights(ABC):

    def __init__(self, nodes: list=None, edges: list=None):
        self.scm = nx.DiGraph()
        self.default_nodes = nodes.copy() if nodes is not None else None
        self.default_edges = edges.copy() if edges is not None else None
        self._init_structure(nodes, edges)
        assert len(list(nx.simple_cycles(self.scm))) == 0

    def _init_structure(self, nodes: list=None, edges: list=None) -> None:
        """
        Initialize the node and edge weights. Moreover, initialize for each
        node its parents.
        """
        self.scm.add_nodes_from(
            nodes if nodes else self.default_nodes
        )

        self.scm.add_edges_from(
            edges if edges else self.default_edges
        )

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

    def set_scm_structure(self, scm_structure: dict={}):
        """Set the SCM structure

        :param scm_structure: dictionary containing the nodes and edges.
        :type scm_structure: dict
        """
        self.scm = nx.DiGraph()
        self._init_structure(
            scm_structure.get("nodes", None),
            scm_structure.get("edges", None)
        )
    
    def reset_scm_to_default(self) -> None:
        """Reset the SCM structure to the default values
        """
        self.set_scm_structure({})