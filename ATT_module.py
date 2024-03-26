from torch import nn
import torch

class ATTLayer(nn.Module):
    def __init__(self, channel, reduction=18):
        super(ATTLayer, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel * 3, bias=False),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x) # (1500,32,1,1)
        y = y.view(b, c) # (1500,32)
        y = self.fc(y) # (1500,32)
        y = y.view(b, c*3, 1, 1) # (1500,32,1,1)
        y1, y2, y3 = torch.split(y, [self.channel, self.channel, self.channel], dim=1)
        return y1,y2,y3