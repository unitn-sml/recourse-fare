from rl_mcts.core.utils.functions import import_dyn_class
from rl_mcts.core.agents.policy_only_agent import PolicyOnly

import numpy as np
import pandas as pd

from argparse import ArgumentParser
import yaml

import time
import os
from tqdm import tqdm

import torch

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the automa model we want to validate.")
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")
    parser.add_argument("--save", default=False, action="store_true", help="Save result to file")
    parser.add_argument("--to-stdout", default=False, action="store_true", help="Print results to stdout")

    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    reward = 0
    results_file = None
    costs = None
    total_actions = None
    length_actions = None
    results_filename = None

    env = import_dyn_class(config.get("environment").get("name"))(
        **config.get("environment").get("configuration_parameters", {}),
        **config.get("validation").get("environment").get("configuration_parameters", {})
    )

    method = "agent_only"
    dataset = config.get("validation").get("dataset_name")

    num_programs = env.get_num_programs()
    observation_dim = env.get_obs_dimension()
    programs_library = env.programs_library

    # Set up the encoder needed for the environment
    encoder = import_dyn_class(config.get("environment").get("encoder").get("name"))(
        env.get_obs_dimension(),
        config.get("environment").get("encoder").get("configuration_parameters").get("encoding_dim")
    )

    additional_arguments_from_env = env.get_additional_parameters()

    policy = import_dyn_class(config.get("policy").get("name"))(
        encoder,
        config.get("policy").get("hidden_size"),
        num_programs,
        config.get("policy").get("encoding_dim"),
        **additional_arguments_from_env
    )

    policy.load_state_dict(torch.load(args.model))

    idx = env.prog_to_idx["INTERVENE"]
    failures = 0

    ts = time.localtime(time.time())
    date_time = '-validation-static-{}_{}_{}-{}_{}_{}.csv'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])

    if args.save:
        results_filename = config.get("validation").get("save_results_name") + date_time
        # results_file = open(
        #    os.path.join(config.get("validation").get("save_results"), results_filename), "w"
        # )

    # perform validation, not training
    env.validation = True

    reward = []
    costs = []
    total_actions = []
    length_actions = []

    iterations = min(int(config.get("validation").get("iterations")), len(env.data))

    for _ in tqdm(range(0, iterations), disable=args.to_stdout):

        idx = env.prog_to_idx["INTERVENE"]

        network_only = PolicyOnly(policy, env, env.max_depth_dict)
        netonly_reward, trace_used, cost = network_only.play(idx)

        reward.append(netonly_reward)
        if netonly_reward > 0:
            total_actions.append(trace_used)
            length_actions.append(len(trace_used))
            costs.append(sum(cost))

    # Create dataframe with the complete actions
    traces = []

    for k, trace in enumerate(total_actions):
        for p, a in trace:
            traces.append([
                k, p, a
            ])

    if len(traces) != 0:
        t = pd.DataFrame(traces, columns=["id", "program", "argument"])

    #print("Correct:", sum(reward))
    #print("Failures:", iterations-sum(reward))
    # print("Mean/std cost: ", sum(costs)/len(costs), np.std(costs))
    # print("Mean/std length actions: ", sum(length_actions) / len(length_actions), np.std(length_actions))

    # Fix if they are empty
    costs = costs if costs else [0]
    length_actions = length_actions if length_actions else [0]

    if args.to_stdout:
        print(
            f"{method},{dataset},{np.mean(reward)},{1 - np.mean(reward)},{np.mean(costs)},{np.std(costs)},{np.mean(length_actions)},{np.std(length_actions)},0.0,0.0")

        if traces:
            # Create a dataframe and save sequences to disk
            best_sequences = pd.DataFrame(traces, columns=["id", "program", "arguments"])
            best_sequences.to_csv(
                os.path.join(config.get("validation").get("save_results"),
                             f"traces-{method}-{dataset}-{results_filename}"),
                index=None)
