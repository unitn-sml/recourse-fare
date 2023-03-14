from rl_mcts.core.utils.functions import import_dyn_class, get_cost_from_tree, get_trace
from rl_mcts.core.data_loader import DataLoader
from rl_mcts.core.mcts.MCTS import MCTS
from rl_mcts.core.buffer.trace_buffer import PrioritizedReplayBuffer
from rl_mcts.core.trainer.trainer import Trainer
from rl_mcts.core.trainer.trainer_statistics import MovingAverageStatistics
from rl_mcts.core.agents.policy import Policy

import torch

from tqdm import tqdm

import pandas as pd
import numpy as np

# The class must take as input:
# - The user features X
#   > Features might be also the user costs A for the causal graph
#   > We might also have a problem to integrate this with the environment
# - The model predictions Y
# - A pre-trained black-box model (it can work with any model)
# - An environment, which specify which actions are available
# - ..?

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

        # Load the policy from a pretrained FARE model
        if policy_config.get("model_path"):
            self.policy.load_state_dict(torch.load(policy_config.get("model_path")))

    def _init_training_objects(self) -> None:

        # Initialize the replay buffer. It is needed to store the various traces for training
        self.buffer = PrioritizedReplayBuffer(self.training_buffer_size,
                                            p1=self.training_buffer_sample_error
                                            )

        # Set up the trainer algorithm
        self.trainer = Trainer(self.policy, self.buffer, MCTS, batch_size=self.batch_size)

        # Set up the curriculum statistics that decides the next experiments to be done
        self.training_statistics = MovingAverageStatistics(moving_average=0.99)

    def save(self, save_model_path="."):
        torch.save(self.policy.state_dict(), save_model_path)

    def predict(self, X, full_output=False, verbose=False):

        X = X.to_dict(orient='records')

        counterfactuals = []
        Y = []
        traces = []
        costs = []
        for i in tqdm(range(len(X)), disable=not verbose):

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

            Y.append(1 if task_reward > 0 else 0)
            counterfactuals.append(env_validation.features.copy())
        
        if full_output:
            return pd.DataFrame.from_records(counterfactuals), Y, traces, costs
        else:
            return pd.DataFrame.from_records(counterfactuals)

    def fit(self, X, max_iter=1000, verbose=False):

        # Initialize the various objects needed to train FARE
        self._init_training_objects()

        for iteration in tqdm(range(1, max_iter+1), desc="Train FARE", disable=verbose):

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
            self.trainer.train_one_step([traces])

            if iteration % self.validation_steps == 0 and verbose:

                self.policy = self.trainer.policy

                _, validation_rewards, traces, costs = self.predict(
                    X.sample(self.trainer.num_validation_episodes),
                    full_output=True)
                lengths = [len(trace) for trace in traces]

                # Update the statistics
                self.training_statistics.update_statistics(validation_rewards, costs, lengths)

                print(f"[*] Iteration {iteration} / Buffer Size: {self.buffer.get_memory_length()} / {self.training_statistics.print_statistics(string_out=True)}")
    
        # Copy the trainer policy to the object policy 
        self.policy = self.trainer.policy