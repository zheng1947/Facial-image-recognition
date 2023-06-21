import torch
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import transforms
from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.distributions as dist
from mlp21 import gMLP
import math
# from model_c2 import Cnn1

from tqdm import tqdm, trange

# 跑单独的x-y
# CNN+MLP混合
batch_size = 80
learning_rate = 0.0001
epochs = 3000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device = torch.device('cpu')#('cuda:0')
version = '0.01'
l = 1  # 厚度
d = 6  # 直径
fr = 18500  # 剩余磁场
u0 = torch.tensor(4 * math.pi * (10 ** -7))
pi = torch.tensor(math.pi)
m1 = (((math.pi) * d * d * l * fr) / (4 * u0))
x = np.load('raw_data.npy')
y = np.load('label.npy')
y = y[:,1:3]

def resample(mean, variance):
    normal = dist.Normal(0, 1)  # 创建均值为0，方差为1的标准正态分布
    samples = normal.sample(mean.shape).to(device)  # 从标准正态分布中抽样，得到与mean相同形状的样本
    resampled = (variance * samples + mean).to(device)  # 将样本乘以方差，再加上平均数
    return resampled


# 计算Loss函数
def logp(u, o, x):
    epsilon = 1e-8  # 用于稳定计算的小常数
    o = torch.clamp(o, epsilon)  # 限制最小值
    p = (1 / (torch.sqrt(torch.tensor(2 * math.pi)) * o))
    y = -((x - u) ** 2) / (2 * o ** 2)
    loss = -torch.log(p * torch.exp(y) + epsilon)

    # loss = -torch.log(p  + epsilon)
    return torch.mean((x - u) ** 2)


l1 = 101
l = np.size(x, 0)
stop_num = 0
stop_check = 7
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)  # float32
len1 = x.shape[0]
x = torch.reshape(x,(len1,16384))
x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)  # float32
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
len2 = x_test.shape[0]
x_test = torch.reshape(x_test,(len2,16384))
len3 = x_train.shape[0]
x_train = torch.reshape(x_train,(len3,16384))
train_dataset = data.TensorDataset(x_train, y_train)
test_dataset = data.TensorDataset(x_test, y_test)

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # shuffle是否对数据进行打乱
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train = list(enumerate(train_dataloader))
test = list(enumerate(test_dataloader))

net_f = gMLP(in_features=16384, out_features=2, mlp_features=256, num_blocks=5).to(device)
net_c = gMLP(in_features=16384, out_features=2, mlp_features=256, num_blocks=5).to(device)
net_f1 = gMLP(in_features=2, out_features=16384, mlp_features=256, num_blocks=5).to(device)
net_c1 = gMLP(in_features=2, out_features=16384, mlp_features=256, num_blocks=5).to(device)
# net_f.load_state_dict(torch.load('finlf4.1.pkl'))
# net_c.load_state_dict(torch.load('finlc4.1.pkl'))
# net_f1.load_state_dict(torch.load('finlf.pkl'))
# net_c1.load_state_dict(torch.load('finlc1.7.pkl'))
optimizer_f = optim.Adam(net_f.parameters(), lr=learning_rate)
optimizer_c = optim.Adam(net_c.parameters(), lr=learning_rate)
optimizer_f1 = optim.Adam(net_f1.parameters(), lr=learning_rate)
optimizer_c1 = optim.Adam(net_c1.parameters(), lr=learning_rate)
criteon = nn.SmoothL1Loss().to(device)

# x_new = x_new.T

the = 0.01
g = 300000
m = 0
losstest = np.array([])
losstrain = np.array([])
losstest_x = np.array([])
losstrain_x = np.array([])
epoch_train = np.array([])
epoch_test = np.array([])
epoch_train_x = np.array([])
epoch_test_x = np.array([])
# predict_the = torch.tensor([200, 200, 200, 200, 200, 200, 200, 200, 200]).to(device)
predict_the = torch.from_numpy(np.zeros(384)).to(device)
predict_the = predict_the.view(1, -1)
predict_the = predict_the.repeat_interleave(batch_size, dim=0)
p_train = 0

epsilon = 1e-8
normal = dist.Normal(0, 1)  # 创建均值为0，方差为1的标准正态分布
# samples = normal.sample(mean.shape).to(device)  # 从标准正态分布中抽样，得到与mean相同形状的样本
# resampled = (variance * samples + mean).to(device)
for epoch in range(epochs):
    for batch_idx in tqdm(range(len(train)), desc='train'):
        _, (data, target) = train[batch_idx]

        data, target = data.to(device), target.to(device)
        ones = torch.from_numpy(np.zeros(384)).to(device)
        total_euler_out = torch.from_numpy(np.zeros((data.size()[0], 96)))
        total_euler_out1 = torch.from_numpy(np.zeros((data.size()[0], 192)))
        fin_out1 = torch.from_numpy(np.zeros((data.size()[0], 5)))

        if data.size()[0] == predict_the.size()[0]:
            # x_input = torch.cat((data, target), dim=1).float()
            p_train = 0
            predict_1 = net_f(data).requires_grad_(True).to(device)
            predict_2 = net_c(data).requires_grad_(True).to(device)

            #位置
            u = predict_1
            o = predict_2
            x = target
            o = torch.clamp(o, epsilon)  # 限制最小值
            p = (1 / (torch.sqrt(torch.tensor(2 * math.pi)) * o))
            y = -((x - u) ** 2) / (2 * o ** 2)
            p_1 = -torch.log(p * torch.exp(y) + epsilon)
            #resample
            samples = normal.sample(u.shape).to(device)  # 从标准正态分布中抽样，得到与mean相同形状的样本
            resampled = (o * samples + u).to(device)
            #磁场
            predict_3 = net_f1(resampled).requires_grad_(True).to(device)
            predict_4 = net_c1(resampled).requires_grad_(True).to(device)

            u = predict_3
            o = predict_4
            x = data
            o = torch.clamp(o, epsilon)  # 限制最小值
            p = (1 / (torch.sqrt(torch.tensor(2 * math.pi)) * o))
            y = -((x - u) ** 2) / (2 * o ** 2)
            p_2 = -torch.log(p * torch.exp(y) + epsilon)

            loss_c = torch.mean(p_1)+ torch.mean(p_2)*the
            # loss_c =  criteon(predict_1[:,0:5],target)

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            optimizer_f1.zero_grad()
            optimizer_c1.zero_grad()
            loss_c.backward(retain_graph=True)

            optimizer_f.step()
            optimizer_c.step()
            optimizer_f1.step()
            optimizer_c1.step()

            total_loss_x = loss_c
            losstrain_x = np.append(losstrain_x, total_loss_x.item())

        # 最后的batch不足batch_size时
        else:

            predict_1 = net_f(data).requires_grad_(True).to(device)
            predict_2 = net_c(data).requires_grad_(True).to(device)

            u = predict_1
            o = predict_2
            x = target
            o = torch.clamp(o, epsilon)  # 限制最小值
            p = (1 / (torch.sqrt(torch.tensor(2 * math.pi)) * o))
            y = -((x - u) ** 2) / (2 * o ** 2)
            p_1 = -torch.log(p * torch.exp(y) + epsilon)
            #reshample
            samples = normal.sample(u.shape).to(device)  # 从标准正态分布中抽样，得到与mean相同形状的样本
            resampled = (o * samples + u).to(device)
            #磁场
            predict_3 = net_f1(resampled).requires_grad_(True).to(device)
            predict_4 = net_c1(resampled).requires_grad_(True).to(device)
            u = predict_3
            o = predict_4
            x = data
            o = torch.clamp(o, epsilon)  # 限制最小值
            p = (1 / (torch.sqrt(torch.tensor(2 * math.pi)) * o))
            y = -((x - u) ** 2) / (2 * o ** 2)
            p_2 = -torch.log(p * torch.exp(y) + epsilon)
            loss_c = torch.mean(p_1)+ torch.mean(p_2)*the
            # loss_c =  criteon(predict_1[:,0:5],target)

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            optimizer_f1.zero_grad()
            optimizer_c1.zero_grad()

            loss_c.backward(retain_graph=True)

            optimizer_f.step()
            optimizer_c.step()
            optimizer_f1.step()
            optimizer_c1.step()

            total_loss_x = loss_c
            losstrain_x = np.append(losstrain_x, total_loss_x.item())

    epoch_losstrain_x = np.mean(losstrain_x)
    epoch_train_x = np.append(epoch_train_x, epoch_losstrain_x)

    losstrain = np.array([])
    losstrain_x = np.array([])

    print('Train Epoch: {}:  Loss:{:6f}'.format(
        epoch, epoch_losstrain_x
    ))
    test_loss = 0
    for batch_idx in tqdm(range(len(test)), desc='test'):
        _, (test_data, test_target) = test[batch_idx]
        test_data, test_target = test_data.to(device), test_target.to(device)

        if test_data.size()[0] == predict_the.size()[0]:

            predict_1 = net_f(test_data).requires_grad_(True).to(device)
            # predict_2 = net_c(predict_1[:,0:5]).requires_grad_(True).to(device)

            total_loss_x = criteon(predict_1, test_target)
            losstest_x = np.append(losstrain_x, total_loss_x.item())

        else:

            predict_1 = net_f(test_data).requires_grad_(True).to(device)

            total_loss_x = criteon(predict_1, test_target)
            losstest_x = np.append(losstrain_x, total_loss_x.item())

    epoch_losstest_x = np.mean(losstest_x)
    epoch_test_x = np.append(epoch_test_x, epoch_losstest_x)

    losstest = np.array([])
    losstest_x = np.array([])
    if epoch_losstest_x < g:
        torch.save(net_f.state_dict(), 'finlf' + version + '.pkl')
        torch.save(net_c.state_dict(), 'finlc' + version + '.pkl')
        torch.save(net_f1.state_dict(), 'finlf1' + version + '.pkl')
        torch.save(net_c1.state_dict(), 'finlc1' + version + '.pkl')
        g = epoch_losstest_x
        stop_num = 0
        print('较优解：' + str(g))
    else:
        stop_num = stop_num + 1
        print('stop_num' + str(stop_num))

    # if stop_num >= stop_check:
    #     print("Validation loss hasn't improved for {} epochs, stopping training.".format(stop_num))
    #     break
    print('Test set : Averge loss: {:.4f}\n'.format(
        epoch_losstest_x
    ))

with open('位置误差kk1train' + version + '.txt', 'w') as f:
    np.savetxt(f, epoch_train_x)  # np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
    f.close()
with open('位置误差kk1test' + version + '.txt', 'w') as f:
    np.savetxt(f, epoch_test_x)  # np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
    f.close()
# with open('磁场误差train'+version+'.txt', 'a+') as f:
#     np.savetxt(f,epoch_train)#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
#     f.close()
# with open('磁场误差test'+version+'.txt', 'a+') as f:
#     np.savetxt(f,epoch_test)#np.savetxt(r'test.txt', x, fmt='%d', newline='-|-')
#     f.close()
