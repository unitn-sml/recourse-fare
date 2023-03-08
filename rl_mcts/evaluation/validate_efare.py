from rl_mcts.core.utils.functions import import_dyn_class, get_cost_from_env

import numpy as np
import pandas as pd

from argparse import ArgumentParser
import yaml

import time
import os
from tqdm import tqdm

from sklearn.tree import _tree

import dill

def extract_rule_from_tree(model, instance):

    feature = model.tree_.feature
    threshold = model.tree_.threshold

    feature_name = [
        instance.columns[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in feature
    ]

    node_indicator = model.decision_path(instance.values.tolist())
    leaf_id = model.apply(instance.values.tolist())

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
        negation = "" if instance[feature_name[node_id]].values[0] else "not"

        rules_detected.append(
            #f"{feature_name[node_id]} = {inst_value} {threshold_sign} {threshold[node_id]}"
            f"{negation} {feature_name[node_id]}".strip()
        )

    return rules_detected

def validation_recursive_tree(model, env, action, depth, cost, action_list, rules):
    if action == "STOP(0)":
        return [[True, env.memory.copy(), cost, action_list, rules]]
    elif depth < 0:
        return [[False, env.memory.copy(), cost, action_list, rules]]
    else:
        node_name = action.split("(")[0]
        actions = model.get(node_name)

        if isinstance(actions, type(lambda x:0)):
            next_op = actions(None)
            rules.append(["True"])
        else:
            obs_inst = pd.DataFrame([env.get_observation().tolist()], columns=env.get_observation_columns())
            rules.append(extract_rule_from_tree(actions, obs_inst))
            next_op = actions.predict([env.get_observation().tolist()])[0]

        if next_op != "STOP(0)":
            action_name, args = next_op.split("(")[0], next_op.split("(")[1].replace(")", "")

            action_list.append((action_name, args))

            if args.isnumeric():
                args = int(args)

            precondition_satisfied = True
            if not env.prog_to_precondition.get(action_name)(args):
                precondition_satisfied = False

            if not precondition_satisfied:
                return [[False, env.memory.copy(), cost, action_list, rules]]

            cost += get_cost_from_env(env, action_name, str(args))

            env.act(action_name, args)

            return validation_recursive_tree(model, env, next_op, depth-1, cost, action_list, rules)
        else:

            action_name, args = next_op.split("(")[0], next_op.split("(")[1].replace(")", "")

            action_list.append((action_name, args))

            if args.isnumeric():
                args = int(args)

            cost += get_cost_from_env(env, action_name, str(args))

            return [[True, env.memory.copy(), cost, action_list, rules]]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the automa model we want to validate.")
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")
    parser.add_argument("--single-core", default=True, action="store_false", help="Run everything with a single core.")
    parser.add_argument("--save", default=False, action="store_true", help="Save result to file")
    parser.add_argument("--to-stdout", default=False, action="store_true", help="Print results to stdout")

    args = parser.parse_args()
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)

    if not args.single_core:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        comm = None
        size = 1

    env = None
    reward = 0
    results_file = None
    costs = None
    idusers = None
    total_actions = None
    length_actions = None
    length_rules = None
    total_rules = None
    method = None
    dataset = None
    results_filename = None

    if rank == 0:

        env = import_dyn_class(config.get("environment").get("name"))(
            **config.get("environment").get("configuration_parameters", {}),
            **config.get("validation").get("environment").get("configuration_parameters", {})
        )

        method="program"
        dataset=config.get("validation").get("dataset_name")

        num_programs = env.get_num_programs()
        observation_dim = env.get_obs_dimension()
        programs_library = env.programs_library

        idx_tasks = [prog['index'] for key, prog in env.programs_library.items() if prog['level'] > 0]

        # Set up the encoder needed for the environment
        encoder = import_dyn_class(config.get("environment").get("encoder").get("name"))(
            env.get_obs_dimension(),
            config.get("environment").get("encoder").get("configuration_parameters").get("encoding_dim")
        )

        indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]

        additional_arguments_from_env = env.get_additional_parameters()

        idx = env.prog_to_idx["INTERVENE"]
        failures = 0

        ts = time.localtime(time.time())
        date_time = '-validation-static-{}_{}_{}-{}_{}_{}.csv'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])

        if args.save:
            results_filename = config.get("validation").get("save_results_name")+date_time
            #results_file = open(
            #    os.path.join(config.get("validation").get("save_results"), results_filename), "w"
            #)

        with open(args.model, "rb") as f:
            import dill as pickle
            model = pickle.load(f)

        # perform validation, not training
        env.validation = True

        reward = []
        costs = []
        idusers = []
        total_actions = []
        length_actions = []
        length_rules = []
        total_rules=[]

    iterations = min(int(config.get("validation").get("iterations")), len(env.data))

    for iduser in tqdm(range(0, iterations//size), disable=args.to_stdout):

        if not args.single_core:
            env = comm.bcast(env, root=0)

        idx = env.prog_to_idx["INTERVENE"]

        _, state_index = env.start_task(idx)

        max_depth = env.max_depth_dict.get(1)

        next_action = "INTERVENE(0)"

        results = validation_recursive_tree(model, env, next_action, max_depth, 0, [], [])

        if not args.single_core:
            results = comm.gather(results, root=0)
        else:
            results = [results]

        env.end_task()

        if rank == 0:
            for R in results:
                for r in R:
                    env.memory = r[1]
                    if env.prog_to_postcondition[env.get_program_from_index(idx)](None, None) and r[0]:
                        reward.append(1)
                        idusers.append(iduser)
                        costs.append(r[2])
                        total_actions.append(r[3])
                        length_rules += [len(x) for x in r[4]]
                        total_rules.append(r[4])
                        length_actions.append(len(r[3]))
                        break
                    else:
                        reward.append(0)

    if rank == 0:

        # Create dataframe with the complete actions
        traces = []

        for (k, trace, rules) in zip(idusers,total_actions, total_rules):
            for (p, a), r in zip(trace, rules):
                traces.append([
                    k, p, a, " AND ".join(r)
                ])

        t = pd.DataFrame(traces, columns=["id", "program", "argument", "rule"])

        # Fix if they are empty
        costs = costs if costs else [0]
        length_actions = length_actions if length_actions else [0]

        if args.to_stdout:
            print(f"{method},{dataset},{np.mean(reward)},{1 - np.mean(reward)},{np.mean(costs)},{np.std(costs)},{np.mean(length_actions)},{np.std(length_actions)},0.0,0.0,{np.mean(length_rules)},{np.std(length_rules)}")

        if args.save:
            # Create a dataframe and save sequences to disk
            if traces:
                best_sequences = pd.DataFrame(traces, columns=["id", "program", "arguments", "rule"])
                best_sequences.to_csv(
                    os.path.join(config.get("validation").get("save_results"),
                                     f"traces-{method}-{dataset}-{results_filename}"),
                        index=None)


