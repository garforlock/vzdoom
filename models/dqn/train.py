import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from models.dqn.ma import MA
import torch.optim as optim


def eligibility_trace(batch, dqn):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:

        input = Variable(torch.from_numpy(np.array([series[0].state.numpy(), series[-1].state.numpy()], dtype=np.float32)).unsqueeze(1))
        output = dqn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()

        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward

        state = series[0].state.numpy()
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


def train(dqn, memory, n_steps):
    ma = MA(100)
    loss = nn.MSELoss()
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    nb_epochs = 100

    for epoch in range(1, nb_epochs + 1):
        #memory.run_steps(200)
        for batch in memory.sample_batch(128):
            inputs, targets = eligibility_trace(batch, dqn)
            inputs, targets = Variable(inputs),  Variable(targets.unsqueeze(1))
            predictions = dqn(inputs)
            loss_error = loss(predictions, targets)
            optimizer.zero_grad()
            loss_error.backward()
            optimizer.step()
        rewards_steps = n_steps.rewards_steps()
        print("Rewards: %s", str(rewards_steps))
        ma.add(rewards_steps)
        avg_reward = ma.average()
        print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
        if avg_reward >= 1500:
            print("Congratulations, your AI wins")
            break