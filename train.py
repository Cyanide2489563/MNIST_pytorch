# python version: 3.8
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms

from MLP import MLP

# 定義訓練過程是以 GPU(CUDA) 或 CPU 做運算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 將圖像資料的灰階數值範圍從 0 ~ 255 壓縮成 0 ~ 1 的範圍內
transform = transforms.ToTensor()

# 下載訓練與測試資料集
trainSet = datasets.MNIST(root='dataset', download=True, train=True, transform=transform)
testSet = datasets.MNIST(root='dataset', download=True, train=False, transform=transform)

# 載入訓練與測試資料集
trainLoader = data.DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = data.DataLoader(testSet, batch_size=128, shuffle=False)

# 初始化 NN (Neural Network) 並指定運算裝置
model = MLP().to(device=device)
print("類神經網路結構")
print(model)

# 輸出目前使用的裝置
print("裝置:", list(model.parameters())[0].device)

# 損失函數 (交叉熵)
loss_function = nn.CrossEntropyLoss()
# 指定最佳化方法，採用 SGD(隨機梯度下降)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

valid_loss_min = np.Inf
# 訓練次數
epochs = 50
for epoch in range(epochs):
    # 使模型為訓練模式
    model.train()
    # 訓練損失
    train_loss = 0.0
    # 從資料載入器迭代資料
    for image, label in trainLoader:
        image = image.to(device)    # 將 image 儲存至 GPU
        label = label.to(device)    # 將 label 儲存至 GPU
        # 清除梯度
        optimizer.zero_grad()
        # 將 input (圖像) 傳遞給模型
        output = model(image)
        # 計算損失
        loss = loss_function(output, label)
        # 反向傳播
        loss.backward()
        # 權重更新
        optimizer.step()
        # 更新損失
        train_loss += loss.item() * image.size(0)

    # 使模型為評估模式
    model.eval()
    # 驗證損失
    valid_loss = 0.0
    # 正確分類的數量
    correct = 0
    for image, label in testLoader:
        image = image.to(device)    # 將 image 儲存至 GPU
        label = label.to(device)    # 將 label 儲存至 GPU
        # 將 input (圖像) 傳遞給模型
        output = model(image)
        # 計算損失
        loss = loss_function(output, label)
        # 更新損失
        valid_loss += loss.item() * image.size(0)

        pred = output.argmax(dim=1)
        # 對應位置相等則對應位置爲True,這裏用sum()即記錄了True的數量
        correct += pred.eq(label.data).sum()

    train_loss = train_loss / len(trainLoader.dataset)
    valid_loss = valid_loss / len(testLoader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}\tValidation Loss:{:.6f}'.format(
        epoch + 1,
        train_loss,
        valid_loss
    ))
    # 輸出正確率
    print("正確率: {:.2f}%".format(100. * correct / len(testLoader.dataset)))

    if valid_loss <= valid_loss_min:
        print("Validation los decreased({:.6f}-->{:.6f}). Saving model ..".format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), "model.pt")
        valid_loss_min = valid_loss
