from rl_mcts.core.utils.functions import import_dyn_class, get_cost_from_tree, get_trace
from rl_mcts.core.data_loader import DataLoader
from rl_mcts.core.buffer.trace_buffer import PrioritizedReplayBuffer
from rl_mcts.core.trainer.trainer import Trainer
from rl_mcts.core.trainer.trainer_statistics import MovingAverageStatistics

import torch
import yaml

from tqdm import tqdm

import pandas as pd

class FARE:

    def __init__(self, config, model_path: str=None) -> None:
        
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = yaml.load(open(config),Loader=yaml.FullLoader)

        # Set up the encoder needed for the environment
        self.encoder = import_dyn_class(self.config.get("environment").get("encoder").get("name"))(
            self.config.get("environment").get("encoder").get("configuration_parameters").get("observation_dim"),
            self.config.get("environment").get("encoder").get("configuration_parameters").get("encoding_dim")
        )

        env = import_dyn_class(self.config.get("environment").get("name"))(
            None,None,
            **self.config.get("environment").get("configuration_parameters", {})
        )

        num_programs = env.get_num_programs()
        additional_arguments_from_env = env.get_additional_parameters()

        # Set up the policy object
        self.policy = import_dyn_class(self.config.get("policy").get("name"))(
            self.encoder,
            self.config.get("policy").get("hidden_size"),
            num_programs,
            self.config.get("policy").get("encoding_dim"),
            **additional_arguments_from_env
        )

        # Load the policy from a pretrained FARE model
        if model_path:
            self.policy.load_state_dict(torch.load(model_path))

    def _init_training_objects(self) -> None:

        # Initialize the replay buffer. It is needed to store the various traces for training
        self.buffer = PrioritizedReplayBuffer(self.config.get("training").get("replay_buffer").get("size"),
                                         p1=self.config.get("training").get("replay_buffer").get("sampling_correct_probability")
                                         )

        # Load a pre-trained model to speed up
        if self.config.get("policy").get("pretrained_model", None) is not None:
            self.policy.load_state_dict(torch.load(self.config.get("policy").get("pretrained_model")))

        # Set up the trainer algorithm
        self.trainer = Trainer(self.policy, self.buffer, self.config.get("training").get("mcts").get("name"),
                          batch_size=self.config.get("training").get("trainer").get("batch_size"))

        # Set up the curriculum statistics that decides the next experiments to be done
        self.training_statistics = MovingAverageStatistics(
                                        moving_average=self.config.get("training").get("curriculum_statistics").get("moving_average"))

    def save(self, save_model_path="."):
        torch.save(self.policy.state_dict(), save_model_path)

    def predict(self, X, weights=None, verbose=False, full_output=False):

        X = X.to_dict(orient='records')
        X_w = None
        if weights:
            X_w = weights.to_dict(orient='records')

        counterfactuals = []
        Y = []
        traces = []
        costs = []
        for i in tqdm(range(len(X))):

            features = X[i]
            if X_w:
                weights =  X_w[i]
            else:
                weights = {}

            env_validation = import_dyn_class(self.config.get("environment").get("name"))(
                features.copy(), weights.copy(),
                **self.config.get("environment").get("configuration_parameters", {})
            )
            mcts_validation = import_dyn_class(self.config.get("training").get("mcts").get("name"))(
                env_validation, self.policy,
                **self.config.get("validation").get("mcts").get("configuration_parameters")
            )

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

    def fit(self, X, y, max_iter=None, verbose=False):

        # Initialize the various objects needed to train FARE
        self._init_training_objects()

        dataloader = DataLoader(X=X, y=y, **self.config.get("dataloader").get("configuration_parameters", {}))

        max_iter = self.config.get("training").get("num_iterations") if not max_iter else max_iter

        for iteration in tqdm(range(max_iter), desc="Train FARE", disable=verbose):
            
            for episode in range(self.config.get("training").get("num_episodes_per_iteration")):

                features, weights = dataloader.get_example(sample_errors=0.2)
                env = import_dyn_class(self.config.get("environment").get("name"))(
                    features.copy(),weights.copy(),
                    **self.config.get("environment").get("configuration_parameters", {})
                )
                mcts = import_dyn_class(self.config.get("training").get("mcts").get("name"))(
                    env, self.policy,
                    **self.config.get("training").get("mcts").get("configuration_parameters")
                )

                traces, root_node, node_expanded = mcts.sample_intervention()
                traces = [[features, weights, traces]]
                node_expanded = [node_expanded]

                # Save the failed traces inside the buffer and train only
                # over the successful ones.
                complete_traces = []
                for trace_feature, trace_weights, trace in traces:
                    if trace.task_reward < 0:
                        dataloader.add_failed_example(trace_feature, trace_weights)
                    else:
                        complete_traces.append(trace)

                self.trainer.train_one_step(complete_traces)

            validation_rewards = []
            costs = []
            lengths = []
            for _ in range(self.trainer.num_validation_episodes):

                features, weights = dataloader.get_example()
                env_validation = import_dyn_class(self.config.get("environment").get("name"))(
                    features.copy(),weights.copy(),
                    **self.config.get("environment").get("configuration_parameters", {})
                )
                mcts_validation = import_dyn_class(self.config.get("training").get("mcts").get("name"))(
                    env_validation, self.trainer.policy,
                    **self.config.get("validation").get("mcts").get("configuration_parameters")
                )

                # Sample an execution trace with mcts using policy as a prior
                trace, root_node, _ = mcts_validation.sample_intervention()
                task_reward = trace.task_reward

                cost, _ = get_cost_from_tree(env, root_node)
                costs.append(cost)
                lengths.append(len(trace.previous_actions[1:]))

                validation_rewards.append(task_reward)

            # Update the statistics
            self.training_statistics.update_statistics(validation_rewards, costs, lengths)

            if verbose:
                print(f"[*] Iteration {(iteration+1)*(self.config.get('training').get('num_episodes_per_iteration'))} / Buffer Size: {self.buffer.get_total_successful_traces()} / {self.training_statistics.print_statistics(string_out=True)}")
    
        # Copy the trainer policy to the object policy 
        self.policy = self.trainer.policy