import torch
import pandas as pd

import numpy as np

# Lambda closure, as per
# https://stackoverflow.com/questions/34854400/python-dict-of-lambda-functions
def make_closure(action):
    return lambda x=None: action

class VisualizeAutoma:

    def __init__(self, env, operation="INTERVENE", seed=2021):
        self.env = env
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

        self.encoder = env.data_encoder

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
                    self.env.get_program_from_index(node.program_from_parent_index)
                )

            self.args.append(
                node.args
            )

            self.observations.append(
                self.env.parse_observation(node.env_state)
            )

            self.points.append(
                node.h_lstm.flatten().numpy()
            )

    def compute(self, columns):
        print("[*] Compute rules given graph")

        for p in range(0, len(self.points) - 1):

            ops = (f"{self.operations[p]}({self.args[p]})", f"{self.operations[p + 1]}({self.args[p + 1]})")

            # Get current state
            state = self.operations[p]

            if not state in self.graph:
                self.graph[state] = {"arcs": {}, "data": []}
                self.envs_seen[state] = {}

            if self.real_program_counts[p] < self.real_program_counts[p + 1]:

                self.graph[state]["data"].append(self.observations[p] + [str(ops[1])])

                # Get next state
                next_state = self.operations[p + 1]

                if not next_state in self.graph.get(state):
                    self.graph.get(state)["arcs"] = {ops[1]: set()}
                else:
                    self.graph.get(state)["arcs"][ops[1]] = set()

        for k, v in self.graph.items():
            df = pd.DataFrame(v["data"], columns=columns + ["operation"], dtype=object)
            df.drop_duplicates(inplace=True)
            # Remove inconsistencies
            df = df[df.groupby(columns)["operation"].transform('nunique') == 1]
            df.to_csv(f"{k}.csv", index=None)

            # Stop will be empty, so we do not process
            if k == "STOP":
                continue

            if len(df["operation"].unique()) > 1:
                print(f"[*] Getting rules for node {k}")
                self._compute_tree(f"{k}.csv", k)
            else:
                print(f"[*] Add single rule for node {k}")
                self.graph[k]["arcs"][df["operation"].unique()[0]] = {'True'}

                # If we have the tree then, the operation is simply returning
                # the correct action
                self.automa[k] = make_closure(df["operation"].unique()[0])

    def _compute_tree(self, filename, node_name):

        from sklearn import tree

        df = pd.read_csv(filename)

        Y = df["operation"]
        df.drop(columns=["operation"], inplace=True)

        columns = self.env.categorical_cols
        cat_ohe = self.encoder.transform(df[columns]).toarray()
        ohe_df = pd.DataFrame(cat_ohe, columns=self.encoder.get_feature_names_out(input_features=columns))
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, ohe_df], axis=1).drop(columns=columns, axis=1)

        columns_add = df.columns.tolist()
        columns_add = [c.replace("<=", " ").replace(">=", " ").replace("<", " ").replace(">", " ") for c in columns_add]

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(df.values, Y.values)

        self.automa[node_name] = clf

    def _parse_rule(self, rule):

        body, operation = rule.replace("'", "").replace(": ", "=").split("=>")
        body = " \\n ".join([k.strip() for k in body.split(",")])
        operation = operation.replace("operation=", "").strip()

        return body, operation

    def _convert_bool(self, value):
        if value in ["True", "False"]:
            return value == "True"
        else:
            return value

    def _convert_rule_into_lambda(self, rule):

        body, operation = rule.replace("'", "").split("=>")

        rule_set = []

        rules = body.split(",")
        for r in rules:
            negation = "not" in r
            value = r.split(":")[1].strip()
            feature = r.split(":")[0].replace("not", "").strip()

            value = self._convert_bool(value)

            if negation:
                rule_set.append(lambda x: not (x[feature] == value))
            else:
                rule_set.append(lambda x: x[feature] == value)

        return rule_set

    def _convert_to_dot(self, color="black", dot_file_name=None):

        dot_file_name = "test.dot" if not dot_file_name else dot_file_name

        self.file = open(dot_file_name, 'w')
        self.file.write('digraph g{ \n')

        for node, childs in self.graph.items():

            self.file.write("\t" + str(node) + '\n')

            for child, rules in childs["arcs"].items():

                parsed_rules = " \\n ".join([f"({r})" for r in rules])

                child_name = child.split("(")[0]

                node_rule_name = "\t" + str(child.replace("(", "_").replace(")", "_").replace("/", "_")) + "_" + str(node) + "\t"

                action_rules = node_rule_name
                action_rules += '[ shape=box,'
                if color is not None:
                    action_rules += 'color={}, '.format(color)
                action_rules += 'label=\"{}\"'.format(
                    parsed_rules + "\\n " + child)
                action_rules += '];'

                self.file.write("\t" + action_rules + '\n')

                # Print edge
                res = '{} -> {}'.format(node_rule_name, str(child_name))
                self.file.write("\t" + res + '\n')
                res = '{} -> {} '.format(str(node), node_rule_name)
                self.file.write("\t" + res + '\n')

        self.file.write('}')
        self.file.close()
