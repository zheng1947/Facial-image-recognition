import torch
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import transforms
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import  MinMaxScaler



import math
# from model_c2 import Cnn1

from tqdm import tqdm, trange
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle
from sklearn.datasets import make_classification

import torch
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import transforms
from torch import nn, optim
from matplotlib import pyplot as plt
from tqdm import tnrange
# from model_make import MLP
import os
import math
#用到的与QT通讯的代码

import socket
import threading
from sklearn.preprocessing import  MinMaxScaler
from sklearn.preprocessing import  MinMaxScaler
##主要用于跑结果返回给QT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#

from torch import nn
class gMLPBlock(nn.Module):
    def __init__(self, in_features, mlp_features, dropout_rate=0.0):
        super(gMLPBlock, self).__init__()

        self.mlp_fc1 = nn.Linear(in_features, mlp_features)
        self.mlp_fc2 = nn.Linear(mlp_features, in_features)
        self.gate_fc1 = nn.Linear(in_features, mlp_features)
        self.gate_fc2 = nn.Linear(mlp_features, in_features)
        self.layer_norm1 = nn.LayerNorm(in_features)
        self.layer_norm2 = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        # Compute input pre-activation
        input_norm = self.layer_norm1(x)
        mlp_activations = self.mlp_fc1(input_norm)
        mlp_activations = self.activation(mlp_activations)
        mlp_activations = self.dropout(mlp_activations)
        mlp_activations = self.mlp_fc2(mlp_activations)
        # Compute gate activations
        gate_activations = self.gate_fc1(input_norm)
        gate_activations = self.activation(gate_activations)
        gate_activations = self.dropout(gate_activations)
        gate_activations = self.gate_fc2(gate_activations)
        # Apply gate
        gated_activations = input_norm * torch.sigmoid(gate_activations)
        # Apply activation
        hidden_activations = self.layer_norm2(gated_activations + mlp_activations)
        return hidden_activations

class gMLP(nn.Module):
    def __init__(self, in_features, out_features, mlp_features, num_blocks, dropout_rate=0.0):
        super(gMLP, self).__init__()
        self.input_fc = nn.Linear(in_features, mlp_features)
        self.output_fc = nn.Linear(mlp_features, out_features)
        self.blocks = nn.ModuleList([gMLPBlock(mlp_features, mlp_features, dropout_rate) for i in range(num_blocks)])
        self.activation = nn.GELU()
    def forward(self, x):
        # Compute input pre-activation
        input_activations = self.input_fc(x)
        # Apply activation
        hidden_activations = self.activation(input_activations)
        # Pass through blocks
        for block in self.blocks:
            hidden_activations = block(hidden_activations)
        # Compute output
        output_activations = self.output_fc(hidden_activations)
        return output_activations

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # num_classes是分类问题中的类别数量
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 30 * 30)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 不使用softmax激活函数，保留原始输出
        return x
# x = np.load('test_raw_data.npy')
# y = np.load('test_label.npy')
x = np.load('raw_data.npy')
y = np.load('label.npy')
y = y[:,1:5]
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
len1 = x.shape[0]
# x = torch.reshape(x,(len1,1,128,128))
# net_c = CNN()
# net_c.load_state_dict(torch.load('finlv00.001.pkl'))
# # print(max(y[:,3]))
# output = net_c(x)
# output = torch.round(output)
# print('准确度')
# acc = np.sum(output[:,0].detach().numpy() ==y[:,0].detach().numpy())/len1
# print(acc)
# acc = np.sum(output[:,1].detach().numpy() ==y[:,1].detach().numpy())/len1
# print(acc)
# acc = np.sum(output[:,2].detach().numpy() ==y[:,2].detach().numpy())/len1
# print(acc)
# acc = np.sum(output[:,3].detach().numpy() ==y[:,3].detach().numpy())/len1
# print(acc)
# print('AUC')
# a = output[:,2].detach().numpy()
# b = y[:,2].detach().numpy()
# fpr, tpr, thresholds = roc_curve(b,a)
# roc_auc = auc(fpr, tpr)
# print(fpr)
# print(tpr)
# print(roc_auc)
# # 绘制AUC曲线
# plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
# plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.savefig('CNN.png', dpi=600,format="png")
# plt.show()


x = torch.reshape(x,(len1,16384)) 
net_c1 =  gMLP(in_features=16384,out_features=2, mlp_features=256, num_blocks=5)
# net_c1.load_state_dict(torch.load('finlv01.00.pkl'))
# net_c1.load_state_dict(torch.load('finlv01.pkl'))
# net_c1.load_state_dict(torch.load('finlf0.01.pkl'))
net_c1.load_state_dict(torch.load('finlf0.0006.pkl'))
output = net_c1(x)
output = torch.round(output)
print(y)
# y_input = output.detach().numpy()-y.detach().numpy()

acc = np.sum(output.detach().numpy() ==y.detach().numpy())/len1
print(acc)
acc = np.sum(output[:,1].detach().numpy() ==y[:,1].detach().numpy())/len1
print(acc)
# acc = np.sum(output[:,2].detach().numpy() ==y[:,2].detach().numpy())/len1
# print(acc)
# acc = np.sum(output[:,3].detach().numpy() ==y[:,3].detach().numpy())/len1
# print(acc)

a = output[:,0].detach().numpy()
b = y[:,0].detach().numpy()
fpr, tpr, thresholds = roc_curve(b,a)
roc_auc = auc(fpr, tpr)
print(fpr)
print(tpr)
print(roc_auc)
# 绘制AUC曲线
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('CNN.png', dpi=600,format="png")
plt.show()