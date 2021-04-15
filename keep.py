from DataOperate import MySet,MySet_npy, get_data_list
from FCNet import FCN16s
import time
import os
import numpy as np
import cv2
import SimpleITK as itk
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler

def get_k_fold_data(k, i, X):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = len(X) // k  # 每份的个数:数据总条数/折数（组数）
    train_list, val_list = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # slice(start,end,step)切片函数
        ##idx 为每组 valid
        part = X[idx]
        if j == i:  ###第i折作valid
            val_list = part
        elif train_list is None:
            train_list = part
        else:
            train_list = train_list + part  # dim=0增加行数，竖着连接
    return train_list, val_list
def k_fold(k, X,ids):
    for i in range(k):
        total = 0
        positive = 0
        torch.cuda.empty_cache()
        net = FCN16s(1)
        net = torch.nn.DataParallel(net, ids).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4,weight_decay=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.75)
        train_list, val_lsit = get_k_fold_data(k, i, X)  # 获取k折交叉验证的训练和验证数据
        train_set = MySet_npy(train_list)
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
        val_set = MySet_npy(val_lsit)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        criterion_bce = torch.nn.BCELoss()

        ### 每份数据进行训练,体现步骤三####
        train(i, net, train_loader, val_loader, optimizer, criterion_bce, scheduler)


def train(k, net, train_loader, val_loader, optimizer, criterion_bce):
    for epoch in range(0, 1000):

        print("Epoch: {}".format(epoch))

        for batch_idx, (image,mask,label,name) in enumerate(train_loader):
            print(name)
            start_time = time.time()
            image = Variable(image.cuda())
            mask = Variable(mask.cuda())
            output = net(image)

            optimizer.zero_grad()
            output = F.sigmoid(output)



            loss0_bce = criterion_bce(output[0], mask)


            loss =  loss0_bce
            loss.backward()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_ids = [0]
train_list = get_data_list("/home/ubuntu/liuyiyao/3D_breast_Seg/Dataset/miccai_data_64*256*256_patch", ratio=0.8)
k_fold(5, train_list)

