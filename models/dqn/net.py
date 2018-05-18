from torch import nn
import torch.nn.functional as F


class NET(nn.Module):
    def __init__(self, available_actions_count):
        super(NET, self).__init__()
        #n_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(in_features=192, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=available_actions_count)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x)))
        x = F.relu(F.max_pool2d(self.convolution2(x)))
        count_neurons = x.data.view(1, -1).size(1)
        return count_neurons

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x)