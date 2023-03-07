import numpy as np
import torch
from collections import OrderedDict


class MovingAverageStatistics():

    def __init__(self, programs_library, moving_average=0.95):
        
        self.moving_average = moving_average
        self.maximum_level = 1

        self.task_name = [k for k,p in programs_library.items() if p['level'] == self.maximum_level][0]
        self.task_index = [p['index'] for _,p in programs_library.items() if p['level'] == self.maximum_level][0]

        self.task_average_reward = 0
        self.task_stats_update = 0

    def get_task_index(self):

        return self.task_index

    def print_statistics(self, string_out=False):
        '''
        Print current learning statistics (in terms of rewards).
        '''
        res = '%s: %.3f '% (self.task_name, self.task_average_reward)
        if string_out:
            return res
        print(res)

    def get_statistic(self, task_index):
        """
        Returns the current average reward on the task.
        Args:
            task_index: task to know the statistic
        Returns:
            average reward on this task
        """
        return self.task_average_reward

    def update_statistics(self, rewards):
        """This function must be called every time a new task has been attempted by NPI. It is used to
        update tasks average rewards as well as the maximum_task level.
        Args:
          task_index: the task that has been attempted
          reward: the reward obtained at the end of the task
          rewards:
        """
        # Update task average reward
        for reward in rewards:
            # all non-zero rewards are considered to be 1.0 in the curriculum scheduler
            reward = 1.0 if reward > 0.0 else 0.0
            self.task_average_reward = self.moving_average*self.task_average_reward + (1-self.moving_average)*reward