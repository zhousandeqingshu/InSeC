import shutil

import torch
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import pandas as pd
import os
import numpy as np
import torch.nn as nn
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torch.nn import functional as F
import argparse
from tqdm import tqdm

# from utils.画loss import plot_list

from contextlib import redirect_stdout

# from MYmodel.数据增强 import Feedforward # 导入前馈前连接网络

from 探究注意力机制.注意力库 import SELayer


def calculate_p(a):
    # 设定常量值
    Tmax = 143
    Tmin = 20
    result = []
    for row in a:
        a1 = row[0]
        a2 = row[1]

        if a1 <= 197:
            p1int = ((a1 + 2) // 3) + 19
            p1fra = a1 - p1int * 3 + 58
        else:
            p1int = a1 - 112
            p1fra = 0
        p1min = max(Tmin, p1int - 5)
        p1max = min(Tmax, p1min + 9)
        p1min = p1max - 9

        temp = ((a2 + 2) // 3) - 1
        p2int = temp + p1min
        p2fra = a2 - 2 - temp * 3

        result.append([p1int, p1fra, p2int, p2fra])
    return result


def save_best_model(net, optimizer, epoch, acc, dir):
    state = {
        'epoch': epoch,
        'acc': acc,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        shutil.rmtree(dir)
        os.mkdir(dir)

    filename = os.path.join(dir, 'model_best' + str(epoch + 1) + '.pth')
    torch.save(state, filename)


# def read_csv_file(file_path):
#     return pd.read_table(file_path, header=None, sep=' ').iloc[:, : 3].values


from multiprocessing import Pool, freeze_support

# def getFile(directory_path, sz=-1):
#     file_list = []
#     for root, dirs, files in os.walk(directory_path):
#         for file in files:
#             file_list.append(os.path.join(root, file))
#     if sz > 0:
#         file_list = file_list[:sz]
#     with Pool() as pool:
#         results = list(tqdm(pool.imap(read_csv_file, file_list), total=len(file_list)))
#
#     return results
#
# def split_v(v, split_len):
#     if split_len == 0:
#         return v
#     res = []
#     for _v in v:
#         _x = _v[:split_len]
#         res.append(_x)
#     return res

from concurrent.futures import ThreadPoolExecutor, as_completed  # 多线程

from concurrent.futures import ProcessPoolExecutor, as_completed  # 多进程


def getFile(directory_path, sz=-1):
    file_list = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    if sz > 0:
        file_list = file_list[:sz]

    res = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for file_path in file_list:
            future = executor.submit(process_file, file_path)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files", unit="file"):
            result = future.result()
            res.append(result)
    return res


def process_file(file_path):
    file = pd.read_table(file_path, header=None, sep=' ').values
    file_first_three_cols = file[:, :3]
    file_last_two_cols = file[:, 3:5]
    calculated_cols = calculate_p(file_last_two_cols)
    merged_file = np.concatenate((file_first_three_cols, calculated_cols), axis=1)
    return merged_file


def split_v(v, split_len):
    if split_len == 0:
        return v
    res = []
    for _v in v:
        _x = _v[:split_len]
        res.append(_x)
    return res


def create_batch(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# *********************************************************************************************************
class Feedforward(nn.Module):
    def __init__(self, input_size, emd_size, out_size):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(input_size, emd_size)  # 全连接层1，输入维度为input_size，输出维度为emd_size
        self.fc2 = nn.Linear(emd_size, out_size)  # 全连接层2，输入维度为emd_size，输出维度为out_size

        self.dropout = nn.Dropout(0.5)  # 添加一个dropout层用于数据增强

        if input_size != out_size:  # 如果输入维度和输出维度不相等，则需要添加一个线性层用于维度匹配
            self.shortcut = nn.Linear(input_size, out_size)
        else:
            self.shortcut = nn.Identity()  # 映射

    def forward(self, x):
        residual = x  # 保存输入的残差
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.dropout(x)  # 数据增强
        x = self.fc2(x)

        shortcut = self.shortcut(residual)  # 使用线性层进行维度匹配
        x += shortcut  # 将残差加到输出上
        x = torch.relu(x)  # 使用ReLU激活函数

        return x


class AttentionConvNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionConvNet, self).__init__()
        self.attention = nn.Linear(input_dim, input_dim)
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        # 注意力机制
        attention_weights = torch.softmax(self.attention(x), dim=2)
        attended_x = x * attention_weights

        # 一维卷积
        output = self.conv(attended_x.permute(0, 2, 1))
        output = output.permute(0, 2, 1)

        return output


''''
CBAM模块
'''


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        # self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
        #                       padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        #
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # x = spatial_out * x
        return x


class FeatureLearningNetwork(nn.Module):
    def __init__(self, input_size=3):
        super(FeatureLearningNetwork, self).__init__()
        self.lstm_Q = nn.LSTM(input_size, 50, 1)
        self.lstm_P = nn.LSTM(50, 50, 1)

        self.attetion = AttentionConvNet(input_size,100)

        # CBAM
        # self.attetion = CBAMLayer(1000)
        self.fc1 = nn.Linear(7, 100)

        self.fc = nn.Linear(100, 4)

        self.forwardLinear = Feedforward(input_size, 64, 100)

    def forward(self, x):
        # 为CBAM提供输入数据
        z = x.unsqueeze(0)
        # print('z',z.shape)

        z = torch.transpose(z, 1, 2)

        MQ, _ = self.lstm_Q(x)
        # print('MQ',MQ.shape) #(100,333,50)
        MP, _ = self.lstm_P(MQ)  # (100,333,50)
        # print('MP' ,MP.shape)
        M = torch.cat((MP, MQ), dim=2)  # (100,333,100# )
        t = self.forwardLinear(x)

        e = self.attetion(x)
        # e = self.fc1(e)
        # e = e.squeeze(0)
        # print('e',e.shape)
        # e = torch.transpose(e, 0, 1)

        t = self.forwardLinear(x)
        #
        # print('t',t.shape)
        # print('e',e.shape)
        # print('M',M.shape)

        M += e + t
        # print('M', M.shape)  #(100,333,4)
        M = self.fc(M)
        # print('M', M.shape)  #(100,333,4)
        return M


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()

        # 三层卷积层，保持输入通道数，输出通道数逐渐增加
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Batch Normalization层
        self.batch_norm = nn.BatchNorm1d(64)
        # 全连接层，将输出维度降为1
        self.fc = nn.Linear(64, 32)
        # ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入x的维度为（200, X, 4）
        x = x.permute(0, 2, 1)  # 将维度调整为（200, 4, X）以适应卷积层的输入要求
        print('x.shape',x.shape)
        # 通过三层卷积层
        x = self.relu(self.conv1(x))
        print('x1.shape', x.shape)
        x = self.relu(self.conv2(x))
        print('x2.shape', x.shape)
        x = self.relu(self.conv3(x))
        print('x3.shape', x.shape)

        # Batch Normalization
        x = self.batch_norm(x)
        print('x4.shape',x.shape)
        # 池化操作，可根据实际需要选择不同的池化方法
        x = torch.mean(x, dim=2)  # 在最后一个维度上取平均，维度变为（200, 64）
        print('x5.shape', x.shape)

        # 全连接层得到输出，维度变为（200, 1）
        x = self.fc(x)
        print('x6.shape', x.shape)

        return x


class zh(nn.Module):
    def __init__(self, input_size=3):
        super(zh, self).__init__()
        self.FeatureLearningNetwork = FeatureLearningNetwork(input_size)
        self.slidNet = ConvolutionalNetwork()
        self.sigmod = nn.Sigmoid()

        self.linear = nn.Sequential(
            nn.Linear(32, 16),  nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        out = self.FeatureLearningNetwork(x)
        out = self.slidNet(out)
        out = self.linear(out)
        return out


# *********************************************************************************************************
def evaluate_accuracy_gpu(net, data_iter, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    count_list = [0, 0]
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            y_hat = net(X)
            count_list[0] += calcACC(y_hat, y)
            count_list[1] += y.numel()
    return count_list[0] / count_list[1]


def calcACC(y_hat, y):
    return ((y_hat >= 0.5) == (y >= 0.5)).sum()


# @save
def train_ch6(net, train_iter, test_iter, valid_iter, num_epochs, lr, device, BestParameter):
    """用GPU训练模型(在第六章定义)"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv3d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    loss = nn.BCELoss()
    train_l = 0
    train_acc = 0
    acc_ans = 0
    count_list = [0, 0, 0]

    loss_running = []
    best_accrary = []

    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        net.train()
        for X, y in train_iter:
            X = X.to(device=device, dtype=torch.float)
            optimizer.zero_grad()
            y = y.to(device=device, dtype=torch.float)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                count_list[0] += l * X.shape[0]
                count_list[1] += calcACC(y_hat, y)
                count_list[2] += X.shape[0]
            train_l = count_list[0] / count_list[2]
            train_acc = count_list[1] / count_list[2]
            loss_running.append(train_l)

        import time
        # 在每轮训练后评估模型在测试数据集上的精度
        time_start = time.time()
        valid_acc = evaluate_accuracy_gpu(net, test_iter)
        if valid_acc > acc_ans:
            acc_ans = valid_acc
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            best_accrary.append(round(test_acc.item() * 100, 2))
            print("目前存放了测试集的准确率最大值如下:", max(best_accrary))
        print(epoch + 1, "/", num_epochs, end=' ')
        print(f'loss {train_l:.3f}, train_acc {train_acc * 100:3f}, ans acc {acc_ans * 100:.3f}')
        print(f'{(time.time() - time_start) *1000} ms')
        print(
            f'loss {train_l:.3f}, train_acc {train_acc * 100:3f}, ans acc {acc_ans * 100:.3f},time_sample{ 1000* ((time.time() - time_start) / 6164000) } ms')
    print()
    # print(f'loss {train_l:.3f}, train_acc {train_acc * 100:3f}, ans acc {acc_ans * 100:.3f}')
    return loss_running

def main(SPEECH_LEN=0):
    parser = argparse.ArgumentParser(description='Command-line application')
    parser.add_argument('-c', required=False, help='Input carrier directory', default="H:\\New数据集\\feat\\CNV\\CN\\00")
    parser.add_argument('-s', required=False, help='Input steganography directory',
                        default="H:\\New数据集\\混合数据集\\READY\\PMS_CNV\\CN\\30")
    parser.add_argument('-t', type=int, help='Number of frames to extract', default=0)
    parser.add_argument('-d', type=int, help='The Dimension of data', default=3)
    parser.add_argument('-o', required=False, help='Save file')
    args = parser.parse_args()
    path_train_cover = os.path.join(args.c, "train")
    path_test_cover = os.path.join(args.c, "test")
    path_valid_cover = os.path.join(args.c, "valid")
    path_train_stego = os.path.join(args.s, "train")
    path_test_stego = os.path.join(args.s, "test")
    path_valid_stego = os.path.join(args.s, "valid")
    # 训练集
    train_cover_files = getFile(path_train_cover)
    train_stego_files = getFile(path_train_stego)
    # 测试集
    test_cover_files = getFile(path_test_cover)
    test_stego_files = getFile(path_test_stego)
    # 验证集
    valid_cover_files = getFile(path_valid_cover)
    valid_stego_files = getFile(path_valid_stego)
    SPEECH_LEN = SPEECH_LEN
    train_cf = split_v(train_cover_files, SPEECH_LEN)
    train_sf = split_v(train_stego_files, SPEECH_LEN)
    test_cf = split_v(test_cover_files, SPEECH_LEN)
    test_sf = split_v(test_stego_files, SPEECH_LEN)
    valid_cf = split_v(valid_cover_files, SPEECH_LEN)
    valid_sf = split_v(valid_stego_files, SPEECH_LEN)
    # 得到标签数据
    train_data = np.r_[train_cf, train_sf]
    train_label = np.r_[[[0]] * len(train_cf), [[1]] * len(train_sf)]
    test_data = np.r_[test_cf, test_sf]
    test_label = np.r_[[[0]] * len(test_cf), [[1]] * len(test_sf)]
    valid_data = np.r_[valid_cf, valid_sf]
    valid_label = np.r_[[[0]] * len(valid_cf), [[1]] * len(valid_sf)]
    train_data, train_label, test_data, test_label, valid_data, valid_label = map(torch.Tensor, (
    train_data, train_label, test_data, test_label, valid_data, valid_label))

    print('train_data' , train_data.shape)
    print('test_data',test_data.shape)
    print('valid_data',valid_data.shape)

    train_data_s = TensorDataset(train_data, train_label)
    test_data_s = TensorDataset(test_data, test_label)
    valid_data_s = TensorDataset(valid_data, valid_label)
    train_iter = create_batch(train_data_s, batch_size=256)
    test_iter = create_batch(test_data_s, batch_size=256)
    valid_iter = create_batch(valid_data_s, batch_size=256)

    print('training on', "cuda:0")
    Net = zh(input_size=7)
    # Net = zh()
    loss_running = train_ch6(Net, train_iter, test_iter, valid_iter, 100, 0.001, "cuda:0", BestParameter=None)

    return test_iter,valid_iter,train_iter,Net


'''
检测时间
'''
import time
def time_Consumption(valid_iter,net,Speech_Length):
    total_frames = sum(len(sample) for sample, _ in valid_iter.dataset)
    print('传入的迭代器中封装的个数: ' , len(valid_iter.dataset))
    print('总帧长total_frames' , total_frames)

    net.to("cuda:0")
    # 获取 test_iter 中的第一个批次数据
    first_batch = next(iter(test_iter))

    # 假设数据是 (X, y) 形式
    X, y = first_batch
    X = X.to(device="cuda:0", dtype=torch.float)
    y = y.to(device="cuda:0", dtype=torch.float)
    # 打印第一个批次数据的大小
    print(f'X size: {X.size()}, y size: {y.size()}')

    start = time.time()
    y_hat = net(X)
    end=time.time()
    print(f'耗时: { (end-start)*1000} ms')







# RNN-SM
class RNN_SM(nn.Module):
    def __init__(self , t):
        super(RNN_SM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=50, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=500, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.flatten(x)
        print(x.shape)
        x = self.dense(x)
        x = self.activation(x)
        return x
import time

def time_Consumption1(valid_iter, net, Speech_Length):
    total_frames = sum(len(sample) for sample, _ in valid_iter.dataset)
    print('传入的迭代器中封装的个数: ', len(valid_iter.dataset))
    print('总帧长total_frames: ', total_frames)


    # 获取 test_iter 中的第一个批次数据
    first_batch = next(iter(test_iter))
    # 假设数据是 (X, y) 形式
    X, y = first_batch
    # 打印一个批次数据的大小
    print(f'X size: {X.size()}, y size: {y.size()}')


    net.to("cuda:0")
    total_time = 0
    for batch in valid_iter:
        # Assume data is in (X, y) format
        X, y = batch
        X = X.to(device="cuda:0", dtype=torch.float)
        y = y.to(device="cuda:0", dtype=torch.float)

        start = time.time()
        y_hat = net(X)
        end = time.time()
        batch_time = (end - start) * 1000  # Time in milliseconds
        total_time += batch_time

        # Print the time for the current batch
        print(f'Batch time: {batch_time:.2f} ms')

    avg_time_per_frame = total_time / total_frames
    print(f'Total time: {total_time:.2f} ms')
    print(f'Average time per frame: {avg_time_per_frame:.2f} ms')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
    # test_iter, valid_iter, train_iter,Net = main(SPEECH_LEN=200)
    # time_Consumption1(valid_iter,Net,None)

    # print("=================================================")

