from rl_mcts.core.utils.functions import import_dyn_class, get_cost_from_tree

import numpy as np

class Trainer:

    def __init__(self, policy, buffer, mcts_validation_class, batch_size=50, num_updates_per_episode=5, num_validation_episodes=10):

        self.policy = policy
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_updates_per_episode = num_updates_per_episode
        self.num_validation_episodes = num_validation_episodes

        self.validation_mcts_class = import_dyn_class(mcts_validation_class)

    def perform_validation_step(self, env, task_index):

        validation_rewards = []
        costs = []
        for _ in range(self.num_validation_episodes):

            mcts = self.validation_mcts_class(env, self.policy, task_index, exploration=False,
                                              number_of_simulations=5)

            # Sample an execution trace with mcts using policy as a prior
            trace, root_node = mcts.sample_execution_trace()
            task_reward = trace.task_reward

            cost, length = get_cost_from_tree(env, root_node)
            costs.append(cost)

            validation_rewards.append(task_reward)
        return validation_rewards, np.mean(costs)

    def train_one_step(self, traces):

        actor_losses = 0
        critic_losses = 0
        arguments_losses = 0

        # Loop over the traces and save them
        for t in traces:

            if t.task_reward < 0:
                continue

            if t.clean_sub_execution:
                # Append trace to buffer
                self.buffer.append_trace(t.flatten())
            else:
                # TODO: better logging
                print("Trace has not been stored in buffer.")

        if self.buffer.get_memory_length() > self.batch_size:
            for _ in range(self.num_updates_per_episode):
                batch = self.buffer.sample_batch(self.batch_size)
                if batch is not None:
                    actor_loss, critic_loss, arg_loss, _ = self.policy.train_on_batch(batch, False)
                    actor_losses += actor_loss
                    critic_losses += critic_loss
                    arguments_losses += arg_loss

        return actor_losses/self.num_updates_per_episode, critic_losses/self.num_updates_per_episode, \
            arguments_losses/self.num_updates_per_episode