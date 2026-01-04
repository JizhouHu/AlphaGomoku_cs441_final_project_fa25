import torch
import torch.nn as nn
import config

class GomokuLogisticRegression(nn.Module):
    def __init__(self):
        super(GomokuLogisticRegression, self).__init__()
        self.linear = nn.Linear(config.BOARD_SIZE * config.BOARD_SIZE, config.NUM_CLASSES)

    def forward(self, x):
        return self.linear(x)

class GomokuMLP(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = config.BOARD_SIZE * config.BOARD_SIZE  
        hidden = 1024
        hidden2 = 512
        hidden3 = 256
        output_dim = input_dim 

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            #nn.Dropout(0.2),

            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden3, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    

class GomokuCNN(nn.Module):
    def __init__(self):
        super().__init__()

        input_channels = 2
        output_dim = config.BOARD_SIZE * config.BOARD_SIZE
        kernel = 3

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=kernel, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=kernel, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=kernel, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(128*config.BOARD_SIZE * config.BOARD_SIZE, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, output_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        
        super().__init__()
        kernel = 3

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel, padding = 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=kernel, padding = 1),
            nn.BatchNorm2d(channels)         
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        out += x
        return self.relu(out)
    

class GomokuResNet(nn.Module):
    def __init__(self, num_channels=2, num_blocks=5):
        super().__init__()
        board = config.BOARD_SIZE
        channels = 128
        output_dim = board * board

        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board * board, output_dim)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.res_blocks(out)
        return self.decoder(out)