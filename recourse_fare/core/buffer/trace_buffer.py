import numpy as np

class PrioritizedReplayBuffer():

    def __init__(self, max_length, p1=0.8):
        self.memory_task = []
        self.max_length = max_length
        self.p1 = p1
        self.batch_length = 0

    def get_memory_length(self):
        return len(self.memory_task)

    def append_trace(self, trace):
        for tuple in trace:
            self.batch_length = len(tuple) # Set the global tuple length, it should be done only once...
            reward = 0 if tuple[4] <= 0.0 else 1
            if reward <= 0:
                continue
            if len(self.memory_task) >= self.max_length:
                del self.memory_task[0]
            self.memory_task.append(tuple)

    def _sample_sub_batch(self, batch_size, memory):
        indices = np.arange(len(memory))
        sampled_indices = np.random.choice(indices, size=batch_size, replace=(batch_size > len(memory)))
        batch = [[] for _ in range(self.batch_length)]
        for i in sampled_indices:
            for k in range(self.batch_length):
                batch[k].append(memory[i][k])
        return batch

    def sample_batch(self, batch_size):
        memory_1 = []
        if len(self.memory_task) > 0:
            memory_1 += self.memory_task

        if len(memory_1) == 0:
            return None
        elif len(memory_1) > 0:
            batch = self._sample_sub_batch(batch_size, memory_1)

        return batch if batch else None

    def empty_memory(self):
        self.memory_task = []