from ..utils.functions import import_dyn_class, get_cost_from_tree

import numpy as np

class Trainer:

    def __init__(self, policy, buffer, batch_size=50, num_updates_per_episode=5):

        self.policy = policy
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_updates_per_episode = num_updates_per_episode

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