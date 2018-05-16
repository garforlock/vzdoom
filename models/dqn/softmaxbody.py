import torch

import torch.nn as nn
import torch.nn.functional as F


class SoftmaxBody(nn.Module):

    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T, dim=1)
        m = torch.distributions.Categorical(probs)
        actions = m.sample()
        return actions
