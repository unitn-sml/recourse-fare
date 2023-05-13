import torch
import pandas as pd

import numpy as np

import sklearn
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

from sklearn.compose import make_column_selector

from ..utils.functions import compute_intervention_cost

# Lambda closure, as per
# https://stackoverflow.com/questions/34854400/python-dict-of-lambda-functions
def make_closure(action):
    return lambda x=None: action

class WEFAREModel:

    def __init__(self):
        self.real_program_counts = []
        self.observations = []
        self.points = []
        self.operations = []
        self.args = []
        self.operation = "INTERVENE"

        self.graph = {}
        self.graph_rules = {}
        self.automa = {}
        self.envs_seen = {}

    def get_breadth_first_nodes(self, root_node):
        '''
        Performs a breadth first search inside the tree.

        Args:
            root_node: tree root node

        Returns:
            list of the tree nodes sorted by depths
        '''
        nodes = []
        stack = [root_node]
        while stack:
            cur_node = stack[0]
            stack = stack[1:]
            nodes.append(cur_node)
            for child in cur_node.childs:
                stack.append(child)
        return nodes

    def add(self, root_node, weights, env):

        counter = 0
        stack = [root_node]
        while stack:
            cur_node = stack[0]
            stack = stack[1:]

            if cur_node.selected:
                self.add_point(cur_node, weights, env)
                counter += 1
                self.real_program_counts.append(counter)

            for child in cur_node.childs:
                stack.append(child)

    def add_point(self, node, weights, env):
         
         with torch.no_grad():
            if node.program_from_parent_index is None:
                self.operations.append(
                    self.operation
                )
            else:
                self.operations.append(
                   node.program_name
                )

            self.args.append(
                node.args
            )

            self.points.append(
                node.h_lstm.flatten().numpy()
            )

            env.features = node.env_state.copy()

            # Compute the action costs and save them
            action_costs = {k : [] for k in env.programs_library.keys() if k != "STOP" and k != "INTERVENE"}

            for a in env.get_current_actions():
                intervention = [(a[0], a[1])]

                cost, _ = compute_intervention_cost(env, node.env_state.copy(),
                                                    intervention=intervention, custom_weights=weights)
                
                if a[0] in action_costs:
                    action_costs[a[0]].append(cost)

            action_costs = {k: np.mean(v) if len(v)> 0 else -1 for k, v in action_costs.items()}
            action_costs = {f"{k}_w": v for k,v in action_costs.items()}

            self.observations.append(
                {**node.env_state.copy(), **action_costs.copy()}
            )

    def compute(self):
        print("[*] Compute rules given graph")

        for p in range(0, len(self.operations) - 1):

            ops = (f"{self.operations[p]}({self.args[p]})", f"{self.operations[p + 1]}({self.args[p + 1]})")

            # Get current state
            state = self.operations[p]

            if not state in self.graph:
                self.graph[state] = {"arcs": {}, "data": []}
                self.envs_seen[state] = {}

            if self.real_program_counts[p] < self.real_program_counts[p + 1]:

                new_obs = self.observations[p].copy()
                new_obs["operation"] = str(ops[1])
                self.graph[state]["data"].append(new_obs)

                # Get next state
                next_state = self.operations[p + 1]

                if not next_state in self.graph.get(state):
                    self.graph.get(state)["arcs"] = {ops[1]: set()}
                else:
                    self.graph.get(state)["arcs"][ops[1]] = set()

        for k, v in self.graph.items():

            df = pd.DataFrame.from_dict(v["data"])
            df.drop_duplicates(inplace=True)

            # Stop will be empty, so we do not process
            if k == "STOP":
                continue
            
            if len(df["operation"].unique()) > 1:
                print(f"[*] Getting rules for node {k}")
                self._compute_tree(df, k)
            else:
                print(f"[*] Add single rule for node {k}")
                self.graph[k]["arcs"][df["operation"].unique()[0]] = {'True'}

                # If we have the tree then, the operation is simply returning
                # the correct action
                self.automa[k] = make_closure(df["operation"].unique()[0])

    def _compute_tree(self, data, node_name):

        # We build a pipeline keeping the values intact, while we normalize the
        # values of the weights to ensure better numerical stability

        Y = data["operation"]
        data.drop(columns=["operation"], inplace=True)

        columns_weights = data.columns.tolist()
        columns_weights = [c for c in columns_weights if c.endswith("_w")]
         
        cat_prepreocessor = ColumnTransformer(
             [('categorical',
               OneHotEncoder(handle_unknown="ignore", sparse=False),
               make_column_selector(dtype_include=[object, 'category'])),
               ('scaler',
               KBinsDiscretizer(10, strategy='uniform', encode="onehot-dense"),
               columns_weights)],
             remainder='passthrough')

        pipe = Pipeline([
                         ('preprocessor', cat_prepreocessor),
                         ('model', tree.DecisionTreeClassifier(class_weight='balanced'))
                         ])

        pipe.fit(data, Y)

        self.automa[node_name] = pipe