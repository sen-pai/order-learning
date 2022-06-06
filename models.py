import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNbase(nn.Module):
    """
    A base feature extractor, params chosen assuming dimensions: batch x 3 x 64 x 64 
    If other dimensions are used, it may not work out of the box
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(F.max_pool2d(x, 2))
        x = self.flatten(x)

        return x


class Comparator(nn.Module):
    """
    Comparator accepts the features of two datapoints x_i and x_j and predicts if x_i is lesser, equal or greater than x_j
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3380 * 2, 500)
        self.fc1_drop = nn.Dropout2d()
        self.fc2 = nn.Linear(500, 50)

        # final layer should have dim = 3 (lesser, equal, greater)
        self.fc3 = nn.Linear(50, 3)

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=1)
        # print(f"concatenated shape: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
