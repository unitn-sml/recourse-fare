from ..models.EFARE import EFARE
from ..models.WFARE import WFARE

from ..automa.wefare import WEFAREModel

from ..utils.functions import import_dyn_class, get_cost_from_env, compute_intervention_cost, randomize_actions
from ..utils.functions import convert_string_to_numeric
from ..environment_w import EnvironmentWeights

from tqdm import tqdm

import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import _tree

class WEFARE(EFARE):

    def __init__(self, fare_model: WFARE) -> None:
        super().__init__(fare_model, None)
        
        self.efare_model = WEFAREModel()
        self.model = self.fare_model.model
        self.environment_config = self.fare_model.environment_config
        self.mcts_config = self.fare_model.mcts_config
    
    def fit(self, X, W, G=None, verbose=True):

        _,Y,_,_, root_nodes = self.fare_model.predict(X, W, G, full_output=True, verbose=verbose)

        X_train = X.to_dict("records")
        W_train = W.to_dict("records")
        
        for reward, feature, weights, root_node in tqdm(list(zip(Y, X_train, W_train, root_nodes)), desc="Building dataset W-EFARE"):

            env: EnvironmentWeights = import_dyn_class(self.environment_config.get("class_name"))(
                features = feature.copy(),
                weights = weights.copy(),
                model = self.model,
                **self.environment_config.get("additional_parameters"),
            )
             
            if reward > 0:
                self.efare_model.add(root_node, weights, env)
        
        self.efare_model.compute()
    
    def predict(self, X, W, G=None, full_output=False, verbose=True):

        X = X.to_dict(orient='records')
        W = W.to_dict(orient='records')

        counterfactuals = []
        Y = []
        traces = []
        costs = []
        rules = []
        for i in tqdm(range(len(X)), desc="Eval W-EFARE", disable=not verbose):

            env_validation = import_dyn_class(self.environment_config.get("class_name"))(
                features=X[i].copy(),
                weights=W[i].copy(),
                model=self.model,
                **self.environment_config.get("additional_parameters"))
            
            # If we have the graph structure, override the preset one. 
            if G is not None:
                env_validation.structural_weights.set_scm_structure(G[i])

            env_validation.start_task()

            max_depth = env_validation.max_intervention_depth
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

    def validation_recursive_tree(self, model, env, action, depth, cost, action_list, rules, deterministic_actions=None, randomize=False):
        
        if len(model) == 0:
            # This happens in case the model is not fit
            return [[False, env.features.copy(), cost, action_list, rules]]
        if action == "STOP(0)":
            return [[True, env.features.copy(), cost, action_list, rules]]
        elif depth < 0:
            return [[False, env.features.copy(), cost, action_list, rules]]
        else:
            node_name = action.split("(")[0]
            actions = model.get(node_name)

            if actions is None:
                return self.validation_recursive_tree(model, env, "INTERVENE(0)", depth-1, cost, action_list, rules, randomize=randomize)

            if isinstance(actions, type(lambda x:0)):
                next_op = actions(None)
                rules.append(["True"])
            else:

                #### COMPUTE MEAN ACTION COSTS
                 # Compute the action costs and save them
                action_costs = {k : [] for k in env.programs_library.keys() if k != "STOP" and k != "INTERVENE"}
                for a in env.get_current_actions():
                    intervention = [(a[0], a[1])]
                    
                    cost, _ = compute_intervention_cost(
                        env, env.get_state().copy(), intervention=intervention
                    )

                    if a[0] in action_costs:
                        action_costs[a[0]].append(cost)
                
                action_costs = {k: np.mean(v) if len(v)> 0 else -1 for k, v in action_costs.items()}
                action_costs = {f"{k}_w": v for k,v in action_costs.items()}
                
                next_state = pd.DataFrame.from_records([{**env.get_state().copy(), **action_costs}])

                if sklearn.__version__ >= "1.0.0":
                    rules.append(self.extract_rule_from_tree(actions, next_state))
                else:
                    raise UserWarning("EFARE rules extraction is disabled. Use a scikit-learn version greater than 1.0.0.")
                
                next_op = actions.predict(
                    next_state
                )[0]

                if randomize:
                    next_op = randomize_actions(actions.classes_, next_op, env)

                # Pick the first deterministic action         
                if deterministic_actions:
                    if depth > depth-len(deterministic_actions):
                        next_op = f"{deterministic_actions[0][0]}({deterministic_actions[0][1]})"

            if next_op != "STOP(0)":
                action_name, args = next_op.split("(")[0], next_op.split("(")[1].replace(")", "")

                args = convert_string_to_numeric(args)

                action_list.append((action_name, args))

                precondition_satisfied = True
                if not env.prog_to_precondition.get(action_name)(args):
                    precondition_satisfied = False

                if not precondition_satisfied:
                    return [[False, env.features.copy(), cost, action_list, rules]]

                cost += get_cost_from_env(env, action_name, args)

                env.act(action_name, args)

                return self.validation_recursive_tree(model, env, next_op, depth-1, cost, action_list, rules)
            else:

                action_name, args = next_op.split("(")[0], next_op.split("(")[1].replace(")", "")

                args = convert_string_to_numeric(args)

                action_list.append((action_name, args))

                cost += get_cost_from_env(env, action_name, args)

                return [[True, env.features.copy(), cost, action_list, rules]]

    def extract_rule_from_tree(self, model, instance):

        instance_new_columns = model.named_steps.get("preprocessor").get_feature_names_out(instance.columns)
        instance_new = pd.DataFrame(model.named_steps.get("preprocessor").transform(instance), columns=instance_new_columns)
        model = model.named_steps.get("model")

        feature = model.tree_.feature
        threshold = model.tree_.threshold

        feature_name = [
            instance_new.columns[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in feature
        ]

        node_indicator = model.decision_path(instance_new.values)
        leaf_id = model.apply(instance_new.values)

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
            if instance_new[feature_name[node_id]].values[0] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"
            
            inst_value = "True" if instance_new[feature_name[node_id]].values[0] else "False"
            #negation = "" if instance[feature_name[node_id]].values[0] else "not"

            rules_detected.append(
                f"{feature_name[node_id]} {threshold_sign} {threshold[node_id]}"
                #f"{negation} {feature_name[node_id]}".strip()
            )

        return rules_detected