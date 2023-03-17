from rl_mcts.core.utils.functions import import_dyn_class, get_cost_from_env
from rl_mcts.core.agents.policy import Policy
from rl_mcts.core.mcts.MCTS import MCTS

from rl_mcts.core.automa.efare import EFAREModel
from rl_mcts.core.models.FARE import FARE

from tqdm.auto import tqdm

import pandas as pd

from sklearn.tree import _tree

class EFARE():

    def __init__(self, fare_model: FARE, preprocessor=None) -> None:

        # Black-box model we want to use
        self.fare_model = fare_model

        # The EFARE model we want to train
        self.efare_model = EFAREModel()
        self.efare_preprocessor = preprocessor

    def load(self, load_path:str = "."):
        with open(load_path, "rb") as f:
            import dill as pickle
            self.efare_model.automa = pickle.load(f)

    def save(self, save_path:str="."):
        with open(save_path, "wb") as f:
            import dill as pickle
            pickle.dump(self.efare_model.automa, f)
    
    def fit(self, X, verbose=True):

        _,Y,_,_, root_nodes = self.fare_model.predict(X, full_output=True, verbose=verbose)

        for reward,root_node in zip(Y,root_nodes):
            if reward > 0:
                self.efare_model.add(root_node)
        
        self.efare_model.compute(self.efare_preprocessor)
    
    def predict(self, X, full_output=False, verbose=True):

        X = X.to_dict(orient='records')

        counterfactuals = []
        Y = []
        traces = []
        costs = []
        rules = []
        for i in tqdm(range(len(X)), desc="Eval EFARE", disable=not verbose):

            env_validation = import_dyn_class(self.fare_model.environment_config.get("class_name"))(
                X[i].copy(),
                self.fare_model.model,
                **self.fare_model.environment_config.get("additional_parameters"))

            env_validation.start_task()

            max_depth = env_validation.max_depth_dict
            next_action = "INTERVENE(0)"

            results = self.validation_recursive_tree(self.efare_model.automa,
                                                     env_validation,
                                                     next_action,
                                                     max_depth, 0, [], [])[0]

            counterfactuals.append(results[1].copy())
            traces.append(results[3])
            Y.append(env_validation.prog_to_postcondition(X[i].copy(), results[1].copy()))
            costs.append(results[2])
            rules.append(results[4])

            env_validation.end_task()
        
        if full_output:
            return pd.DataFrame.from_records(counterfactuals), Y, traces, costs, rules
        else:
            return pd.DataFrame.from_records(counterfactuals)

    def validation_recursive_tree(self, model, env, action, depth, cost, action_list, rules):
        
        if action == "STOP(0)":
            return [[True, env.features.copy(), cost, action_list, rules]]
        elif depth < 0:
            return [[False, env.features.copy(), cost, action_list, rules]]
        else:
            node_name = action.split("(")[0]
            actions = model.get(node_name)

            if isinstance(actions, type(lambda x:0)):
                next_op = actions(None)
                rules.append(["True"])
            else:

                if self.efare_preprocessor:
                    next_state = self.efare_preprocessor.transform(pd.DataFrame.from_records([env.get_state()]))
                    transformed_columns = self.efare_preprocessor.get_feature_names_out(pd.DataFrame.from_records([env.get_state()]).columns)
                    next_state = pd.DataFrame(next_state, columns=transformed_columns)
                else:
                    next_state = pd.DataFrame.from_records([env.get_state()])

                rules.append(self.extract_rule_from_tree(actions, next_state))
                next_op = actions.predict(
                    next_state
                )[0]

            if next_op != "STOP(0)":
                action_name, args = next_op.split("(")[0], next_op.split("(")[1].replace(")", "")

                action_list.append((action_name, args))

                if args.isnumeric():
                    args = int(args)

                precondition_satisfied = True
                if not env.prog_to_precondition.get(action_name)(args):
                    precondition_satisfied = False

                if not precondition_satisfied:
                    return [[False, env.features.copy(), cost, action_list, rules]]

                cost += get_cost_from_env(env, action_name, str(args))

                env.act(action_name, args)

                return self.validation_recursive_tree(model, env, next_op, depth-1, cost, action_list, rules)
            else:

                action_name, args = next_op.split("(")[0], next_op.split("(")[1].replace(")", "")

                action_list.append((action_name, args))

                if args.isnumeric():
                    args = int(args)

                cost += get_cost_from_env(env, action_name, str(args))

                return [[True, env.features.copy(), cost, action_list, rules]]
    
    def extract_rule_from_tree(self, model, instance):

        feature = model.tree_.feature
        threshold = model.tree_.threshold

        feature_name = [
            instance.columns[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in feature
        ]

        node_indicator = model.decision_path(instance)
        leaf_id = model.apply(instance)

        sample_id = 0
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
                    node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                    ]

        rules_detected = []

        for node_id in node_index:

            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            # check if value of the split feature for sample 0 is below threshold
            if instance[feature_name[node_id]].values[0] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"
            
            inst_value = "True" if instance[feature_name[node_id]].values[0] else "False"
            #negation = "" if instance[feature_name[node_id]].values[0] else "not"

            rules_detected.append(
                f"{feature_name[node_id]} {threshold_sign} {threshold[node_id]}"
                #f"{negation} {feature_name[node_id]}".strip()
            )

        return rules_detected