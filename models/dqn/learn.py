import random
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from random import randint, random


class Learn:
    def __init__(self, trainer, model, criterion, optimizer, memory, batch_size, discount_factor):
        self.trainer = trainer
        self.model = model
        self.learning_rate = 0.00025
        self.discount_factor = 0.99
        self.epochs = 20
        self.learning_steps_per_epoch = 2000
        self.replay_memory_size = 10000
        self.frame_repeat = 12
        self.resolution = (30, 45)
        self.episodes_to_watch = 10
        self.criterion = criterion
        self.optimizer = optimizer
        self.memory = memory
        self.batch_size = batch_size
        self.discount_factor = discount_factor

    def learn(self, s1, target_q):
        s1 = torch.from_numpy(s1)
        target_q = torch.from_numpy(target_q)
        s1, target_q = Variable(s1), Variable(target_q)
        output = self.model(s1)
        loss = self.criterion(output, target_q)
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_q_values(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        return self.model(state)

    def learn_from_memory(self):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        if self.memory.size > self.batch_size:
            s1, a, s2, isterminal, r = self.memory.get_sample(self.batch_size)

            q = self.get_q_values(s2).data.numpy()
            q2 = numpy.max(q, axis=1)
            target_q = self.get_q_values(s1).data.numpy()
            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
            target_q[numpy.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q2
            self.learn(s1, target_q)

    def get_best_action(self, state):
        q = self.get_q_values(state)
        m, index = torch.max(q, 1)
        action = index.data.numpy()[0]
        return action

    def exploration_rate(self, epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * self.epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * self.epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    def perform_learning_step(self, epoch):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        s1 = self.trainer.get_screen(self.resolution[0], self.resolution[1])

        # With probability eps make a random action.
        eps = self.exploration_rate(epoch)
        if random() <= eps:
            a = randint(0, len(self.trainer.actions) - 1)
        else:
            # Choose the best action according to the network.
            s1 = s1.reshape([1, 1, self.resolution[0], self.resolution[1]])
            a = self.get_best_action(s1)
        reward = self.trainer.game.make_action(self.trainer.actions[a], self.frame_repeat)

        isterminal = self.trainer.game.is_episode_finished()
        s2 = self.trainer.get_screen(self.resolution[0], self.resolution[1]) if not isterminal else None

        # Remember the transition that was just experienced.
        self.memory.add_transition(s1, a, s2, isterminal, reward)

        self.learn_from_memory()
