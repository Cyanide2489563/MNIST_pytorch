import torch
from torch import nn


# 定義類神經網路
from torchviz import make_dot


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Fully Connected Layer
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        # 定義 Dropout 機率 P = 0.5
        self.dropout = nn.Dropout(0.5)

    # 定義向前傳播函數
    def forward(self, x):
        # 展開輸入
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        x = self.dropout(nn.functional.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# 輸出網路結構圖
_model = MLP()
_input = torch.rand(13, 1, 28, 28)
MyConvNetVis = make_dot(_model(_input), params=dict(_model.named_parameters()))
MyConvNetVis.format = "svg"
MyConvNetVis.directory = "data"
