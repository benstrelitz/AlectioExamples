import torch.nn as nn
import torch.nn.functional as F


class dCNN(nn.Module):
    def __init__(self):
        super(dCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=1, padding=0)

        self.b_n_1 = nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(24, 56, kernel_size=(4, 4), stride=1, padding=0)

        self.b_n_2 = nn.BatchNorm2d(56)

        self.conv3 = nn.Conv2d(56, 120, kernel_size=(4, 4), stride=1, padding=0)

        self.b_n_3 = nn.BatchNorm2d(120)

        self.conv4 = nn.Conv2d(120, 248, kernel_size=(4, 4), stride=1, padding=0)

        self.b_n_4 = nn.BatchNorm2d(248)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.con1 = nn.Linear(in_features=248 * 5 * 8, out_features=120)

        self.con2 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):

        """
        torch.Size([64, 1, 128, 173])
        torch.Size([64, 24, 63, 85])
        torch.Size([64, 56, 30, 41])
        torch.Size([64, 120, 13, 19])
        torch.Size([64, 248, 5, 8])
        """

        s = False

        if s:
            print(x.size())  # -->

        x = self.pool(F.relu(self.b_n_1(self.conv1(x))))

        if s:
            print(x.size())  # -->

        x = self.pool(F.relu(self.b_n_2(self.conv2(x))))

        if s:
            print(x.size())  # -->

        x = self.pool(F.relu(self.b_n_3(self.conv3(x))))

        if s:
            print(x.size())  # -->

        x = self.pool(F.relu(self.b_n_4(self.conv4(x))))

        if s:
            print(x.size())  # -->

        x = x.view(-1, 248 * 5 * 8)
        x = F.relu(self.con1(x))
        x = self.con2(x)
        return x
