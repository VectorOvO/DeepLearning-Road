import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import torch
from numpy import argmax
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import Linear
from torch.onnx.symbolic_opset11 import vstack
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn import MaxPool2d, Conv2d, ReLU, Softmax, CrossEntropyLoss
from torch.nn.init import kaiming_uniform, kaiming_uniform_, xavier_uniform, xavier_uniform_
from torch.nn import Module
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from PIL import Image
import torchvision.transforms as transforms
import torch
from Draw import test_dl
from torch.utils.tensorboard import SummaryWriter


# 定义数据集
class CSVDataset(Dataset):
    # 导入数据集
    def __init__(self, path):
        # 导入传入路径的数据集为 Pandas DataFrame 格式
        df = pd.read_csv(path, header=None)
        # 设置神经网络的输入与输出
        self.X = df.values[:, :-1]  # 根据你的数据集定义输入属性
        self.y = df.values[:, -1]  # 根据你的数据集定义输出属性
        # 确保输入的数据是浮点型
        self.X = self.X.astype('float32')
        # 使用浮点型标签编码原输出
        self.y = LabelEncoder().fit_transform(self.y)

    # 定义获得数据集长度的方法
    def __len__(self):
        return len(self.X)

    # 定义获得某一行数据的方法
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # 在类内部定义划分训练集和测试集的方法，在本例中，训练集比例为 0.67，测试集比例为 0.33
    def get_splits(self, n_test=0.33):
        # 确定训练集和测试集的尺寸
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # 根据尺寸划分训练集和测试集并返回
        return random_split(self, [train_size, test_size])
#模型创建
# 模型定义
class CNN(Module):
    # 定义模型属性
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        # 输入到隐层 1
        self.hidden1 = Conv2d(n_channels, 32, (3, 3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # 池化层 1
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))
        # 隐层 2
        self.hidden2 = Conv2d(32, 32, (3, 3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # 池化层 2
        self.pool2 = MaxPool2d((2, 2), stride=(2, 2))
        # 全连接层
        self.hidden3 = Linear(5 * 5 * 32, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # 输出层
        self.hidden4 = Linear(100, 10)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

    # 前向传播
    def forward(self, X):
        # 输入到隐层 1
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # 隐层 2
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # 扁平化
        X = X.view(-1, 5 * 5 * 32)
        # 隐层 3
        X = self.hidden3(X)
        X = self.act3(X)
        # 输出层
        X = self.hidden4(X)
        X = self.act4(X)
        return X

# 准备数据集
def prepare_data(path):
    # 定义标准化
    trans = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    # 加载数据集
    train = MNIST(path, train=True, download=True, transform=trans)
    test = MNIST(path, train=False, download=True, transform=trans)
    # 为训练集和测试集创建 DataLoader
    train_dl = DataLoader(train, batch_size=64, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

writer = SummaryWriter(log_dir='runs/mnist_experiment')

    # 训练模型
def train_model(train_dl,model,device):
    # 定义优化器
    model.to(device)
    criterion=CrossEntropyLoss()
    optimizer=SGD(model.parameters(),lr=0.001,momentum=0.9)
    writer=SummaryWriter(log_dir="runs/mnist_experiment")# 创建日志目录
    # 枚举 epochs
    for epoch in range(30): # 20轮训练
        running_loss = 0.0
        total,correct = 0,0
        # 枚举 mini batches
        for i,(inputs,targets) in enumerate(train_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            # 梯度清楚
            optimizer.zero_grad()
            outputs = model(inputs)
            # 计算模型输出
            yhat=model(inputs)
            # 计算损失
            loss=criterion(yhat,targets)
            # 贡献度分配
            loss.backward()
            # 升级模型权重
            optimizer.step()

            running_loss+=loss.item()
            preds=torch.argmax(yhat,dim=1)
            correct+=(preds==targets).sum().item()
            total+=targets.size(0)
        avg_loss=running_loss/len(train_dl)
        acc=correct/total
        print(f"Epoch(第n次迭代) {epoch + 1}, Loss(此时的损失): {avg_loss:.4f}, Accuracy(模型精确度): {acc * 100:.2f}%")


        # 每轮记录到 TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
    writer.close()
    print("训练日志已写入 TensorBoard")


# 评估模型
def evaluate_model(test_dl, model, device='cpu'):
    model.to(device)
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_dl:
            inputs = inputs.to(device)
            yhat = model(inputs)                      # logits
            preds = torch.argmax(yhat, dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()
            predictions.append(preds)
            actuals.append(targets_np)
    # 合并所有批次
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    acc = accuracy_score(actuals, predictions)
    return acc
# 准备数据
# path = '~/.torch/datasets/mnist'
# train_dl, test_dl = prepare_data(path)
# print(len(train_dl.dataset), len(test_dl.dataset))
# # GPU加速
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 定义网络
# model = CNN(1)
# # 训练模型
# train_model(train_dl, model)  # 该步骤运行约需 5 分钟。
# # 保存模型
# torch.save(model.state_dict(), "mnist_cnn.pth")
#
# # 评估模型
# acc = evaluate_model(test_dl, model)
# print('Accuracy: %.3f' % acc)
if __name__ == "__main__":
    path = './mnist_data'
    model_file = "mnist_cnn.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(1)

    if os.path.exists(model_file):
        print(f"加载已有模型: {model_file}")
        model.load_state_dict(torch.load(model_file, map_location=device))
        acc = evaluate_model(test_dl, model)
        acc *= 100
        print('该模型的准确率为: %.2f%%' % acc)
    else:
        print("没有找到模型，开始训练...")
        train_dl, test_dl = prepare_data(path)
        train_model(train_dl, model, device)
        torch.save(model.state_dict(), model_file)
        acc = evaluate_model(test_dl, model)
        acc*=100
        print('该模型的准确率为: %.2f%%' % acc)
        print(f"训练完成，模型已保存到 {model_file}")
