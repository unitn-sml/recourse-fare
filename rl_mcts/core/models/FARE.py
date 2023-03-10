from rl_mcts.core.utils.functions import import_dyn_class, get_cost_from_tree, get_trace
from rl_mcts.core.data_loader import DataLoader
from rl_mcts.core.buffer.trace_buffer import PrioritizedReplayBuffer
from rl_mcts.core.trainer.trainer import Trainer
from rl_mcts.core.trainer.trainer_statistics import MovingAverageStatistics

import torch
import yaml

from tqdm import tqdm

class FARE:

    def __init__(self, config, model_path: str=None) -> None:
        
        self.config = yaml.load(open(config),Loader=yaml.FullLoader)

        # Initialize the various objects needed to train FARE
        self._init_training_objects()

        # Load the policy from a pretrained FARE model
        if model_path:
            self.policy.load_state_dict(torch.load("model_path"))

    def _init_training_objects(self) -> None:

        self.dataloader = DataLoader(**self.config.get("dataloader").get("configuration_parameters", {}))
        f,w = self.dataloader.get_example()

        env = import_dyn_class(self.config.get("environment").get("name"))(
            f,w,
            **self.config.get("environment").get("configuration_parameters", {})
        )

        num_programs = env.get_num_programs()
        programs_library = env.programs_library

        idx_tasks = [prog['index'] for key, prog in env.programs_library.items() if prog['level'] > 0]

        # Initialize the replay buffer. It is needed to store the various traces for training
        self.buffer = PrioritizedReplayBuffer(self.config.get("training").get("replay_buffer").get("size"),
                                         idx_tasks,
                                         p1=self.config.get("training").get("replay_buffer").get("sampling_correct_probability")
                                         )

        # Set up the encoder needed for the environment
        self.encoder = import_dyn_class(self.config.get("environment").get("encoder").get("name"))(
            env.get_obs_dimension(),
            self.config.get("environment").get("encoder").get("configuration_parameters").get("encoding_dim")
        )

        additional_arguments_from_env = env.get_additional_parameters()

        self.policy = import_dyn_class(self.config.get("policy").get("name"))(
            self.encoder,
            self.config.get("policy").get("hidden_size"),
            num_programs,
            self.config.get("policy").get("encoding_dim"),
            **additional_arguments_from_env
        )

        # Load a pre-trained model to speed up
        if self.config.get("policy").get("pretrained_model", None) is not None:
            self.policy.load_state_dict(torch.load(self.config.get("policy").get("pretrained_model")))

        # Set up the trainer algorithm
        self.trainer = Trainer(self.policy, self.buffer, self.config.get("training").get("mcts").get("name"),
                          batch_size=self.config.get("training").get("trainer").get("batch_size"))

        # Set up the curriculum statistics that decides the next experiments to be done
        self.training_statistics = MovingAverageStatistics(programs_library,
                                        moving_average=self.config.get("training").get("curriculum_statistics").get("moving_average"))

    def save(self, save_model_path="."):
        torch.save(self.trainer.policy.state_dict(), save_model_path)

    def predict(self, X, verbose=False, full_output=False):

        task_index = self.training_statistics.get_task_index()

        counterfactuals = []
        Y = []
        traces = []
        costs = []
        for i in range(len(X)):

            features, weights = X[i]

            env_validation = import_dyn_class(self.config.get("environment").get("name"))(
                features.copy(),weights.copy(),
                **self.config.get("environment").get("configuration_parameters", {})
            )
            mcts_validation = import_dyn_class(self.config.get("training").get("mcts").get("name"))(
                env_validation, self.trainer.policy, task_index,
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
            return counterfactuals, Y, traces, costs
        else:
            return counterfactuals


    def fit(self, X=None, max_iter=None, verbose=False):

        task_index = self.training_statistics.get_task_index()
        
        max_iter = self.config.get("training").get("num_iterations") if not max_iter else max_iter

        for iteration in tqdm(range(max_iter), desc="Train FARE", disable=verbose):
            
            for episode in range(self.config.get("training").get("num_episodes_per_iteration")):

                features, weights = self.dataloader.get_example(sample_errors=0.2)
                env = import_dyn_class(self.config.get("environment").get("name"))(
                    features.copy(),weights.copy(),
                    **self.config.get("environment").get("configuration_parameters", {})
                )
                mcts = import_dyn_class(self.config.get("training").get("mcts").get("name"))(
                    env, self.trainer.policy, task_index,
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
                        self.dataloader.add_failed_example(trace_feature, trace_weights)
                    else:
                        complete_traces.append(trace)

                self.trainer.train_one_step(complete_traces)

            # Get id of the current 
            task_index = self.training_statistics.get_task_index()

            validation_rewards = []
            costs = []
            lengths = []
            for _ in range(self.trainer.num_validation_episodes):

                features, weights = self.dataloader.get_example()
                env_validation = import_dyn_class(self.config.get("environment").get("name"))(
                    features.copy(),weights.copy(),
                    **self.config.get("environment").get("configuration_parameters", {})
                )
                mcts_validation = import_dyn_class(self.config.get("training").get("mcts").get("name"))(
                    env_validation, self.trainer.policy, task_index,
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