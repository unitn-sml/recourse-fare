import torch
import pandas as pd

import numpy as np

import sklearn

# Lambda closure, as per
# https://stackoverflow.com/questions/34854400/python-dict-of-lambda-functions
def make_closure(action):
    return lambda x=None: action

class EFAREModel:

    def __init__(self, operation="INTERVENE", seed=2021):
        
        self.real_program_counts = []
        self.observations = []
        self.points = []
        self.operations = []
        self.args = []
        self.operation = operation
        self.seed = seed

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

    def add(self, root_node):

        counter = 0
        stack = [root_node]
        while stack:
            cur_node = stack[0]
            stack = stack[1:]

            if cur_node.selected:
                self.add_point(cur_node)
                counter += 1
                self.real_program_counts.append(counter)

            for child in cur_node.childs:
                stack.append(child)

    def add_point(self, node):

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

            self.observations.append(
               node.env_state
            )

            self.points.append(
                node.h_lstm.flatten().numpy()
            )

    def compute(self, preprocessor=None):
        print("[*] Compute rules given graph")

        for p in range(0, len(self.points) - 1):

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
            df = pd.DataFrame.from_records(v["data"])
            df.drop_duplicates(inplace=True)
            # Remove inconsistencies
            #df = df[df.groupby(df.columns)["operation"].transform('nunique') == 1]

            # Stop will be empty, so we do not process
            if k == "STOP":
                continue

            if len(df["operation"].unique()) > 1:
                print(f"[*] Getting rules for node {k}")
                self._compute_tree(df, k, preprocessor)
            else:
                print(f"[*] Add single rule for node {k}")
                self.graph[k]["arcs"][df["operation"].unique()[0]] = {'True'}

                # If we have the tree then, the operation is simply returning
                # the correct action
                self.automa[k] = make_closure(df["operation"].unique()[0])

    def _compute_tree(self, df, node_name, preprocessor=None):

        from sklearn import tree

        Y = df["operation"]
        df.drop(columns=["operation"], inplace=True)

        if preprocessor:
            if sklearn.__version__ >= "1.0.0":
                transformed_columns = preprocessor.get_feature_names_out(df.columns)
                df = pd.DataFrame(preprocessor.transform(df), columns=transformed_columns)
            else:
                df = preprocessor.transform(df)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(df, Y.values)

        self.automa[node_name] = clf
