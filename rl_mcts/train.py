from rl_mcts.core.data_loader import DataLoader
from rl_mcts.core.buffer.trace_buffer import PrioritizedReplayBuffer
from rl_mcts.core.trainer.trainer import Trainer
from rl_mcts.core.trainer.trainer_statistics import MovingAverageStatistics
from rl_mcts.core.utils.functions import import_dyn_class, get_cost_from_tree

import torch
import numpy as np
import random

from tensorboardX import SummaryWriter

from argparse import ArgumentParser
import yaml
import time
import os

from rl_mcts.core.utils.early_stopping import EarlyStopping

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the file with the experiment configuration")
    parser.add_argument("--single-core", default=False, action="store_true", help="Run everything with a single core.")

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

    task_index = None
    policy = None
    buffer = None
    mcts = None
    trainer = None
    env = None
    statistics = None
    writer = None
    early_stopping = None
    early_stopping_reached = False
    bcast_data = None

    seed = config.get("general").get("seed", 0)

    random.seed(seed+rank)
    np.random.seed(seed+rank)
    torch.manual_seed(seed+rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    MCTS_CLASS = import_dyn_class(config.get("training").get("mcts").get("name"))

    ts = time.localtime(time.time())
    date_time = '{}_{}_{}-{}_{}_{}.model'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
    save_model_path = os.path.join(config.get("general").get("save_model_dir"),
                                   config.get("general").get("save_model_name")+"-"+date_time)

    if rank == 0:

        dataloader = DataLoader(**config.get("dataloader").get("configuration_parameters", {}))
        f,w = dataloader.get_example()

        env = import_dyn_class(config.get("environment").get("name"))(
            f,w,
            **config.get("environment").get("configuration_parameters", {})
        )

        num_programs = env.get_num_programs()
        observation_dim = env.get_obs_dimension()
        programs_library = env.programs_library

        idx_tasks = [prog['index'] for key, prog in env.programs_library.items() if prog['level'] > 0]

        # Initialize the replay buffer. It is needed to store the various traces for training
        buffer = PrioritizedReplayBuffer(config.get("training").get("replay_buffer").get("size"),
                                         idx_tasks,
                                         p1=config.get("training").get("replay_buffer").get("sampling_correct_probability")
                                         )

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

        # Load a pre-trained model to speed up
        if config.get("policy").get("pretrained_model", None) is not None:
            policy.load_state_dict(torch.load(config.get("policy").get("pretrained_model")))

        print(policy)

        # Set up the trainer algorithm
        trainer = Trainer(policy, buffer, config.get("training").get("mcts").get("name"),
                          batch_size=config.get("training").get("trainer").get("batch_size"))

        # Set up the curriculum statistics that decides the next experiments to be done
        statistics = MovingAverageStatistics(programs_library,
                                        moving_average=config.get("training").get("curriculum_statistics").get("moving_average"))

        writer = SummaryWriter(config.get("general").get("tensorboard_dir"))

        early_stopping = EarlyStopping(patience=config.get("training").get("patience", 10), verbose=True,
                                       validation_thresh=config.get("training").get("early_stopping_accuracy", 0.90),
                                       path=save_model_path)

    for iteration in range(config.get("training").get("num_iterations")):

        if rank==0:
            task_index = statistics.get_task_index()
            bcast_data = [task_index, dataloader, trainer.policy, early_stopping_reached]

        act_loss_total = []
        crit_loss_total = []
        args_loss_total = []
        total_node_expanded = []

        for episode in range(config.get("training").get("num_episodes_per_iteration")):

            if not args.single_core:
                task_index, dataloader, policy, early_stopping_reached = comm.bcast(bcast_data, root=0)

            features, weights = dataloader.get_example(sample_errors=0.2)
            env = import_dyn_class(config.get("environment").get("name"))(
                features.copy(),weights.copy(),
                **config.get("environment").get("configuration_parameters", {})
            )
            mcts = MCTS_CLASS(
                env, policy, task_index,
                **config.get("training").get("mcts").get("configuration_parameters")
            )

            # Kill everything if we reached the earlystopping criterion
            if early_stopping_reached:
                break

            traces, root_node, node_expanded = mcts.sample_execution_trace()
            traces = [features, weights, traces]

            if not args.single_core:
                traces = comm.gather(traces, root=0)
                node_expanded = comm.gather(node_expanded, root=0)
            else:
                traces = [traces]
                node_expanded = [node_expanded]

            if rank == 0:

                # Save the failed traces inside the buffer and train only
                # over the successful ones.
                complete_traces = []
                for trace_feature, trace_weights, trace in traces:
                    if trace.task_reward < 0:
                        dataloader.add_failed_example(trace_feature, trace_weights)
                    else:
                        complete_traces.append(trace)

                act_loss, crit_loss, args_loss = trainer.train_one_step(complete_traces)

                act_loss_total.append(act_loss)
                crit_loss_total.append(crit_loss)
                args_loss_total.append(args_loss)
                total_node_expanded += node_expanded

        # Kill everything if we reached the earlystopping criterion
        # This is needed to avoid to hang child processes
        if early_stopping_reached:
            break

        if rank == 0:

            # Get id of the current 
            task_index = statistics.get_task_index()

            v_task_name = env.get_program_from_index(task_index)
            writer.add_scalar("loss/" + v_task_name + "/actor", np.mean(act_loss_total), iteration)
            writer.add_scalar("loss/" + v_task_name + "/value", np.mean(crit_loss_total), iteration)
            writer.add_scalar("loss/" + v_task_name + "/arguments", np.mean(args_loss_total), iteration)

            writer.add_scalar("mcts/" + v_task_name + "/avg_node_expanded", np.mean(total_node_expanded), iteration)
            
            # Print on tensorboard additionals metrics if there are
            for k in env.custom_tensorboard_metrics:
                writer.add_scalar("custom/" + v_task_name + f"/{k}", env.custom_tensorboard_metrics.get(k), iteration)

            validation_rewards = []
            costs = []
            lengths = []
            for _ in range(trainer.num_validation_episodes):

                features, weights = dataloader.get_example()
                env_validation = import_dyn_class(config.get("environment").get("name"))(
                    features.copy(),weights.copy(),
                    **config.get("environment").get("configuration_parameters", {})
                )
                mcts_validation = MCTS_CLASS(
                    env_validation, trainer.policy, task_index,
                    **config.get("validation").get("mcts").get("configuration_parameters")
                )

                # Sample an execution trace with mcts using policy as a prior
                trace, root_node, _ = mcts_validation.sample_execution_trace()
                task_reward = trace.task_reward

                cost, _ = get_cost_from_tree(env, root_node)
                costs.append(cost)
                lengths.append(len(trace.previous_actions[1:]))

                validation_rewards.append(task_reward)

            # Update the statistics
            statistics.update_statistics(validation_rewards, costs, lengths)

            # Get moving average of the statistic
            avg_validity, avg_cost, avg_length = statistics.get_statistic()

            # Get information about early stopping condition
            #early_stopping(np.mean(costs), avg_cost, trainer.policy)
            #if early_stopping.early_stop:
            #    early_stopping_reached = True

            # record on tensorboard
            v_task_name = env.get_program_from_index(task_index)
            writer.add_scalar('validation/' + v_task_name + '/avg_validity', avg_validity, iteration)
            writer.add_scalar('validation/' + v_task_name + '/avg_cost', avg_cost, iteration)
            writer.add_scalar('validation/' + v_task_name + '/avg_length', avg_length, iteration)

            print(f"[*] Iteration {(iteration+1)*(config.get('training').get('num_episodes_per_iteration'))} / Buffer Size: {buffer.get_total_successful_traces()} / {statistics.print_statistics(string_out=True)}")

            # Save policy
            # We save the model only when we reach a satisfactory accuracy
            if config.get("general").get("save_model"):
                torch.save(trainer.policy.state_dict(), save_model_path)


