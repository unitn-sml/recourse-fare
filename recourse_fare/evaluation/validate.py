from ..core.utils.functions import import_dyn_class, get_cost_from_tree, get_trace
from ..core.data_loader import DataLoader

import numpy as np

from argparse import ArgumentParser
import torch
import yaml

import time
import os
from tqdm import tqdm

import random

import pandas as pd

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the model we want to visualize.")
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")
    parser.add_argument("--save", default=False, action="store_true", help="Save result to file")
    parser.add_argument("--output", type=str, help="Override file name output.")
    parser.add_argument("--to-stdout", default=False, action="store_true", help="Print results to stdout")

    args = parser.parse_args()
    config = yaml.load(open(args.config),Loader=yaml.FullLoader)

    seed = config.get("general").get("seed", 0)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataloader = DataLoader(**config.get("validation").get("dataloader").get("configuration_parameters", {}))
    f,w = dataloader.get_example()

    env = import_dyn_class(config.get("environment").get("name"))(
        f,w,
        **config.get("environment").get("configuration_parameters", {}),
        **config.get("validation").get("environment", {}).get("configuration_parameters", {})
    )

    method="mcts"
    dataset=config.get("validation").get("dataset_name")

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

    MCTS_CLASS = import_dyn_class(config.get("training").get("mcts").get("name"))

    # Get the idx of the target class
    idx = None
    for k,v in env.programs_library.items():
        if v.get("level") > 0:
            idx = env.prog_to_idx[k]
    assert idx != None, "Error! The environment has not task with a level greater than 0."

    idusers = []
    mcts_rewards_normalized = []
    mcts_rewards = []
    mcts_cost = []
    mcts_length = []
    best_sequences = []
    custom_metrics = {k:[] for k in env.custom_tensorboard_metrics}
    failures = 0.0

    ts = time.localtime(time.time())
    date_time = '-{}-{}_{}_{}-{}_{}.csv'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])

    results_file = None
    results_filename=None
    if args.save:
        results_filename = config.get("validation").get("save_results_name")+date_time

    iterations = min(int(config.get("validation").get("iterations")), len(dataloader.data))
    for iduser in tqdm(range(0, iterations), disable=args.to_stdout):

        f,w = dataloader.get_example(specific_idx=iduser)

        env = import_dyn_class(config.get("environment").get("name"))(
            f,w,
            **config.get("environment").get("configuration_parameters", {}),
            **config.get("validation").get("environment", {}).get("configuration_parameters", {})
        )
        mcts = MCTS_CLASS(
            env, policy, idx,
            **config.get("validation").get("mcts").get("configuration_parameters")
        )

        trace, root_node, _ = mcts.sample_intervention()

        if trace.rewards[0] > 0:
            cost, length = get_cost_from_tree(env, root_node)
            idusers.append(iduser)
            mcts_rewards.append(trace.rewards[0].item())
            mcts_rewards_normalized.append(1.0)
            mcts_cost.append(cost)
            mcts_length.append(length)
            best_sequences.append(get_trace(env, root_node))
        else:
            mcts_rewards.append(0.0)
            mcts_rewards_normalized.append(0.0)
            failures += 1

        for k in env.custom_tensorboard_metrics:
            custom_metrics[k].append(env.custom_tensorboard_metrics.get(k, 0))

    mcts_rewards_normalized_mean = np.mean(np.array(mcts_rewards_normalized))
    mcts_rewards_normalized_std = np.std(np.array(mcts_rewards_normalized))
    mcts_rewards_mean = np.mean(np.array(mcts_rewards))
    mcts_rewards_std = np.std(np.array(mcts_rewards))
    mcts_cost_mean = np.mean(mcts_cost)
    mcts_cost_std = np.std(mcts_cost)
    mcts_length_mean = np.mean(mcts_length)
    mcts_length_std = np.std(mcts_length)

    complete = f"{mcts_rewards_normalized_mean},{1-mcts_rewards_normalized_mean},{mcts_cost_mean},{mcts_cost_std},{mcts_length_mean},{mcts_length_std}"

    # Custom metric string
    cst_complete = ""
    for k in env.custom_tensorboard_metrics:
        cst_complete += f"{np.mean(custom_metrics.get(k))},{np.std(custom_metrics.get(k))},"
    cst_complete = cst_complete[:-1]  # Remove last comma

    if args.to_stdout:
        print(f"{method},{dataset},{mcts_rewards_normalized_mean},{1-mcts_rewards_normalized_mean},{mcts_cost_mean},{mcts_cost_std},{mcts_length_mean},{mcts_length_std},{cst_complete}")

    # Save results to a file
    if args.save:

        # Save sequences to file
        df_sequences = []
        for (k, x) in zip(idusers, best_sequences):
            for p, a in x:
                df_sequences.append([k, p, a])

        # Create a dataframe and save sequences to disk
        if df_sequences:

            file_name_output = f"traces-{method}-{dataset}-{results_filename}" if not args.output else args.output

            best_sequences = pd.DataFrame(df_sequences, columns=["id", "program", "arguments"])
            best_sequences.to_csv(
                os.path.join(config.get("validation").get("save_results"), file_name_output),
                index=None)
