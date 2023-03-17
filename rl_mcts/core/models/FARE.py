from rl_mcts.core.utils.functions import import_dyn_class, get_cost_from_tree, get_trace
from rl_mcts.core.data_loader import DataLoader
from rl_mcts.core.mcts.MCTS import MCTS
from rl_mcts.core.buffer.trace_buffer import PrioritizedReplayBuffer
from rl_mcts.core.trainer.trainer import Trainer
from rl_mcts.core.trainer.trainer_statistics import MovingAverageStatistics
from rl_mcts.core.agents.policy import Policy

from tensorboardX import SummaryWriter

import torch

from tqdm.auto import tqdm

import pandas as pd
import numpy as np

DEFAULT_MCTS_CONFIG = {
        "exploration": True,
        "number_of_simulations": 10,
        "dir_epsilon": 0.03,
        "dir_noise": 0.3,
        "level_closeness_coeff": 3.0,
        "level_0_penalty": 1.0,
        "qvalue_temperature": 1.0,
        "temperature": 1.3,
        "c_puct": 0.5,
        "gamma": 0.97
    }

DEFAULT_ENVIRONMENT_CONFIG = {
    "class_name": "",
    "additional_parameters": {}
}

DEFAULT_POLICY_CONFIG = {
    "observation_dim": 10,
    "encoding_dim": 10,
    "hidden_size": 40,
    "model_path": None
}

class FARE:

    def __init__(self, model, policy_config=DEFAULT_POLICY_CONFIG,
                 environment_config=DEFAULT_ENVIRONMENT_CONFIG,
                 mcts_config=DEFAULT_MCTS_CONFIG,
                 batch_size=50,
                 training_buffer_size=200,
                 sample_error_probab=0.1,
                 validation_steps = 10) -> None:

        # Black-box model we want to use
        self.model = model

        assert batch_size <= training_buffer_size, "The batch size must be smaller than the training buffer!"

        self.batch_size = batch_size
        self.training_buffer_size = training_buffer_size
        self.training_buffer_sample_error = sample_error_probab

        self.mcts_config = mcts_config
        self.environment_config = environment_config

        self.validation_steps = validation_steps

        env = import_dyn_class(environment_config.get("class_name"))(None, None,
                                                                     **self.environment_config.get("additional_parameters"))

        num_programs = env.get_num_programs()
        additional_arguments_from_env = env.get_additional_parameters()

        # Set up the policy object
        self.policy = Policy(
            policy_config.get("observation_dim"),
            policy_config.get("encoding_dim"),
            policy_config.get("hidden_size"),
            num_programs,
            **additional_arguments_from_env
        )

    def _init_training_objects(self) -> None:

        # Initialize the replay buffer. It is needed to store the various traces for training
        self.buffer = PrioritizedReplayBuffer(self.training_buffer_size,
                                            p1=self.training_buffer_sample_error
                                            )

        # Set up the trainer algorithm
        self.trainer = Trainer(self.policy, self.buffer, batch_size=self.batch_size)

        # Set up the curriculum statistics that decides the next experiments to be done
        self.training_statistics = MovingAverageStatistics(moving_average=0.99)

    def save(self, save_model_path :str="."):
        torch.save(self.policy.state_dict(), save_model_path)

    def load(self, load_model_path :str=".") -> None:
        """Load a pretrained FARE model from a file.

        :param load_model_path: path to the file, defaults to "."
        :type load_model_path: str, optional
        """
        self.policy.load_state_dict(torch.load(load_model_path))

    def predict(self, X, full_output=False, verbose=True):

        X = X.to_dict(orient='records')

        counterfactuals = []
        Y = []
        traces = []
        costs = []
        root_nodes = []
        for i in tqdm(range(len(X)),  desc="Eval FARE", disable=not verbose):

            env_validation = import_dyn_class(self.environment_config.get("class_name"))(
                X[i].copy(),
                self.model,
                **self.environment_config.get("additional_parameters"))
            mcts_validation = MCTS(
                env_validation, self.policy,
                **self.mcts_config
            )

            mcts_validation.exploration = False
            mcts_validation.number_of_simulations = 5

            # Sample an execution trace with mcts using policy as a prior
            trace, root_node, _ = mcts_validation.sample_intervention()
            task_reward = trace.task_reward

            cost, _ = get_cost_from_tree(env_validation, root_node)
            costs.append(cost)
            traces.append(get_trace(env_validation, root_node))
            root_nodes.append(root_node)

            Y.append(1 if task_reward > 0 else 0)
            counterfactuals.append(env_validation.features.copy())
        
        if full_output:
            return pd.DataFrame.from_records(counterfactuals), Y, traces, costs, root_nodes
        else:
            return pd.DataFrame.from_records(counterfactuals)

    def fit(self, X, max_iter=1000, verbose=True, tensorboard=None):

        # Initialize the various objects needed to train FARE
        self._init_training_objects()

        # If tensorboard
        if tensorboard:
            writer = SummaryWriter(tensorboard)

        # Dictionary containing the tensorboard values
        training_losses = {
            "actor": [],
            "arguments": [],
            "value": [],
            "total_nodes": []
        }

        with tqdm(range(1, max_iter+1), desc="Train FARE", disable=not verbose) as t:
            for iteration in t:

                t.set_description(
                    f"Train (Acc={float(self.training_statistics.print_statistics(string_out=True)):.3f}/Buff={self.buffer.get_memory_length()})"
                )

                features = X.sample(1)
                features = features.to_dict(orient='records')[0]

                mcts = MCTS(
                    import_dyn_class(self.environment_config.get("class_name"))(
                        features.copy(),
                        self.model,
                        **self.environment_config.get("additional_parameters")
                        ), 
                    self.policy,
                    **self.mcts_config
                )

                traces, root_node, node_expanded = mcts.sample_intervention()

                # Run one optimization step within the trainer
                act_loss, crit_loss, args_loss = self.trainer.train_one_step([traces])

                training_losses.get("actor").append(act_loss)
                training_losses.get("value").append(crit_loss)
                training_losses.get("arguments").append(args_loss)
                training_losses.get("total_nodes").append(node_expanded)

                # Perform the validation step
                if iteration % self.validation_steps == 0:

                    self.policy = self.trainer.policy

                    _, validation_rewards, traces, costs, _ = self.predict(
                        X.sample(10),
                        full_output=True,
                        verbose=False)
                    lengths = [len(trace) for trace in traces]

                    # Update the statistics
                    self.training_statistics.update_statistics(validation_rewards, costs, lengths)

                    # If tensorboard, update tensorboard files
                    if tensorboard:
                        
                        # Add some information when training
                        writer.add_scalar("loss/actor", np.mean(training_losses.get("actor")), iteration)
                        writer.add_scalar("loss/value", np.mean(training_losses.get("value")), iteration)
                        writer.add_scalar("loss/arguments", np.mean(training_losses.get("arguments")), iteration)
                        writer.add_scalar("mcts/avg_node_expanded", np.mean(training_losses.get("total_nodes")), iteration)

                        # Add some information for the validation
                        avg_validity, avg_cost, avg_length = self.training_statistics.get_statistic()
                        writer.add_scalar('validation/avg_validity', avg_validity, iteration)
                        writer.add_scalar('validation/avg_cost', avg_cost, iteration)
                        writer.add_scalar('validation/avg_length', avg_length, iteration)

                    # Clean the logging
                    for k in training_losses.keys():
                        training_losses[k] = []
    
        # Copy the trainer policy to the object policy 
        self.policy = self.trainer.policy