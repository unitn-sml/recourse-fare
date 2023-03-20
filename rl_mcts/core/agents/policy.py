import torch
import torch.nn as nn
from torch.nn import Linear, LSTMCell, Module
from torch.nn.init import uniform_
import torch.nn.functional as F
import numpy as np

from ..utils.anomaly_detection import BetterAnomalyDetection

class PolicyEncoder(nn.Module):
    """ This class implements a simple MLP which encodes the action policies,
    given the output from the controller. 
    """

    def __init__(self, observation_dim, encoding_dim=20):
        super(PolicyEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, encoding_dim)
        self.l2 = nn.Linear(encoding_dim, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x

class CriticNet(Module):
    def __init__(self, hidden_size):
        super(CriticNet, self).__init__()
        self.l1 = Linear(hidden_size, hidden_size//2)
        self.l2 = Linear(hidden_size//2, 1)

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = torch.tanh(self.l2(x))
        return x

class ActorNet(Module):
    def __init__(self, hidden_size, num_programs):
        super(ActorNet, self).__init__()
        self.l1 = Linear(hidden_size, hidden_size//2)
        self.l2 = Linear(hidden_size//2, num_programs)

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = F.softmax(self.l2(x), dim=-1)
        return x

class ArgumentsNet(Module):

    def __init__(self, hidden_size, args_types_available):
        """
        args_type_available = {
            "INT": list(range(0,100)),
            "CAT": ["A", "B", "C"]
        }

        :param hidden_size: size of the hidden arguments networks
        :param args_types_available: dictionary with all available arguments
        """
        super(ArgumentsNet, self).__init__()

        self.arguments = nn.ModuleList()
        for type_idx, a in args_types_available.items():
            self.arguments.append(ArgumentsSingleNet(hidden_size, argument_ranges=a))

    def forward(self, x):
        """
        Predict the next arguments, concatenate the results
        :param x: input
        :return: distribution probability over the arguments
        """
        results = []
        for i, l in enumerate(self.arguments):
            results.append(self.arguments[i](x))
        return torch.cat(results, 1)


class ArgumentsSingleNet(Module):
    """
    Class which models a single argument network
    """

    def __init__(self, hidden_size: int, argument_ranges: list=None):
        """

        :param hidden_size: size of the hidden linear state. The size is
        cut in half at each layer (e.g., 8 --> 4 --> len(argument_ranges))
        :param argument_ranges: list which contains the total possible arguments
        """

        super(ArgumentsSingleNet, self).__init__()
        if argument_ranges is None:
            argument_ranges = [1,2,3]
        self.l1 = Linear(hidden_size, hidden_size//2)
        self.l2 = Linear(hidden_size//2, len(argument_ranges))

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = F.softmax(self.l2(x), dim=-1)
        return x


class Policy(Module):
    def __init__(self, observation_dim, encoding_dim, hidden_size, num_programs,
                argument_types, learning_rate=1e-3, use_gpu=False):

        super().__init__()

        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"

        self._uniform_init = (-0.1, 0.1)

        self._hidden_size = hidden_size
        self.num_programs = num_programs
        self.argument_types = argument_types

        self.encoding_dim = encoding_dim

        # Initialize networks
        self.encoder = PolicyEncoder(observation_dim, encoding_dim)
        self.encoder = self.encoder.to(self.device)

        self.lstm = LSTMCell(self.encoding_dim, self._hidden_size).to(self.device)
        self.lstm_args = LSTMCell(self.encoding_dim, self._hidden_size).to(self.device)
        self.critic = CriticNet(self._hidden_size).to(self.device)
        self.actor = ActorNet(self._hidden_size, self.num_programs).to(self.device)

        # Generate N decoders given the types and the type range
        self.arguments = ArgumentsNet(self._hidden_size, self.argument_types).to(self.device)

        self.init_networks()
        self.init_optimizer(lr=learning_rate)

        self.args_criterion = torch.nn.BCELoss()

        # Small epsilon used to preven torch.log() to produce nan
        self.epsilon = np.finfo(np.float32).eps

    def init_networks(self):

        for p in self.encoder.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.lstm.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.lstm_args.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.critic.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.actor.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.arguments.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

    def init_optimizer(self, lr):
        '''Initialize the optimizer.
        Args:
            lr (float): learning rate
        '''
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def predict_on_batch(self, e_t, h_t, c_t, h_t_args, c_t_args):
        """Run one NPI inference.
        Args:
          e_t: batch of environment observation
          i_t: batch of calling program
          h_t: batch of lstm hidden state
          c_t: batch of lstm cell state
        Returns:
          probabilities over programs, value, new hidden state, new cell state
        """
        batch_size = len(h_t)
        s_t = self.encoder(e_t.view(batch_size, -1))

        new_h, new_c = self.lstm(s_t, (h_t, c_t))
        new_h_args, new_c_args = self.lstm_args(s_t, (h_t_args, c_t_args))

        actor_out = self.actor(new_h)
        critic_out = self.critic(new_h)

        args_out = self.arguments(new_h_args)

        return actor_out, critic_out, args_out, new_h, new_c, new_h_args, new_c_args

    def train_on_batch(self, batch, check_autograd=False):
        """perform optimization step.
        Args:
          batch (tuple): tuple of batches of environment observations, calling programs, lstm's hidden and cell states
        Returns:
          policy loss, value loss, total loss combining policy and value losses
        """
        e_t = torch.FloatTensor(np.stack(batch[0]))
        lstm_states = batch[2]
        lstm_states_args = batch[6]
        h_t, c_t = zip(*lstm_states)
        h_t, c_t = torch.squeeze(torch.stack(list(h_t))), torch.squeeze(torch.stack(list(c_t)))

        h_t_args, c_t_args = zip(*lstm_states_args)
        h_t_args, c_t_args = torch.squeeze(torch.stack(list(h_t_args))), torch.squeeze(torch.stack(list(c_t_args)))

        policy_labels = torch.squeeze(torch.stack(batch[3]))
        value_labels = torch.stack(batch[4]).view(-1, 1)

        # Use a better anomaly detection
        with BetterAnomalyDetection(check_autograd):

            self.optimizer.zero_grad()
            policy_predictions, value_predictions, args_predictions, _, _, _, _ = self.predict_on_batch(e_t, h_t, c_t, h_t_args, c_t_args)

            policy_loss = -torch.mean(
                policy_labels[:, 0:self.num_programs] * torch.log(policy_predictions + self.epsilon), dim=-1
            ).mean()

            args_loss = -torch.mean(
                policy_labels[:, self.num_programs:] * torch.log(args_predictions + self.epsilon), dim=-1
            ).mean()

            value_loss = torch.pow(value_predictions - value_labels, 2).mean()

            total_loss = (policy_loss + args_loss + value_loss)/3

            total_loss.backward()
            self.optimizer.step()

        return policy_loss.item(), value_loss.item(), args_loss.item(), total_loss.item()

    def forward_once(self, e_t, h, c, h_args, c_args):
        """Run one NPI inference using predict.
        Args:
          e_t: current environment observation
          h: previous lstm hidden state
          c: previous lstm cell state
        Returns:
          probabilities over programs, value, new hidden state, new cell state, a program index sampled according to
          the probabilities over programs)
        """
        e_t = torch.FloatTensor(e_t)
        e_t, h, c, h_args, c_args = e_t.view(1, -1), h.view(1, -1), c.view(1, -1), h_args.view(1, -1), c_args.view(1, -1)
        with torch.no_grad():
            e_t = e_t.to(self.device)
            actor_out, critic_out, args_out, new_h, new_c, new_h_args, new_c_args = self.predict_on_batch(e_t, h, c, h_args, c_args)
        return actor_out, critic_out, args_out, new_h, new_c, new_h_args, new_c_args

    def init_tensors(self):
        """Creates tensors representing the internal states of the lstm filled with zeros.

        Returns:
            instantiated hidden and cell states
        """
        h = torch.zeros(1, self._hidden_size)
        c = torch.zeros(1, self._hidden_size)
        h, c = h.to(self.device), c.to(self.device)

        h_args = torch.zeros(1, self._hidden_size)
        c_args = torch.zeros(1, self._hidden_size)
        h_args, c_args = h_args.to(self.device), c_args.to(self.device)

        return h, c, h_args, c_args