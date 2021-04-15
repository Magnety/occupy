
from DataOperate import MySet ,MySet_npy, get_data_list
from FCNet import FCN16s
import time
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import lr_scheduler
import smtplib
from email.mime.text import MIMEText
from email.header import Header


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
        torch.cuda.empty_cache()
        net = FCN16s(1)
        net = torch.nn.DataParallel(net, ids).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4 ,weight_decay=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.75)
        train_list, val_lsit = get_k_fold_data(k, i, X)  # 获取k折交叉验证的训练和验证数据
        train_set = MySet(train_list)
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
        val_set = MySet(val_lsit)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        criterion_bce = torch.nn.BCELoss()

        ### 每份数据进行训练,体现步骤三####
        train(i, net, train_loader, val_loader, optimizer, criterion_bce)


def train(k, net, train_loader, val_loader, optimizer, criterion_bce):
    for epoch in range(0, 10000):

        print("Epoch: {}".format(epoch))

        for batch_idx, (image ,mask ,label ,name) in enumerate(train_loader):
            print(name)
            start_time = time.time()
            image = Variable(image.cuda())
            mask = Variable(mask.cuda())
            output = net(image)
            optimizer.zero_grad()
            output = F.sigmoid(output)
            loss0_bce = criterion_bce(output, mask)
            loss =  loss0_bce
            loss.backward()


from email.mime.multipart import MIMEMultipart

smtpserver = 'smtp.qq.com'
username = 'mc-yao@qq.com'
password = 'auxwnwzdrtlfbihb'
sender = 'mc-yao@qq.com'
receiver = ['liuyiyao0916@163.com']

import pynvml
import os
import time
train_list = get_data_list("/home/ubuntu/liuyiyao/Data/Breast_s", ratio=0.8)
pynvml.nvmlInit()
devicen_num = pynvml.nvmlDeviceGetCount()
while True:
    for i in range(devicen_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        power_state = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_perf = pynvml.nvmlDeviceGetPerformanceState(handle)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pids = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        print("gpu:", i, '  mem_used:', meminfo.used / 1024 / 1024, '  power:', power_state / 1000, '  perf:',
              power_perf)
        if meminfo.used/1024/1024 < 5000 and power_perf == 8:

            os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%i

            device_ids = [0]
            subject = 'Server 172.21.141.14, GPU : %s is free '%i
            msg = MIMEMultipart('mixed')
            msg['Subject'] = subject
            msg['From'] = 'mc-yao@qq.com <mc-yao@qq.com>'
            msg['To'] = ";".join(receiver)
            text = 'Server 172.21.141.14, GPU : %s is free '%i
            text_plain = MIMEText(text, 'plain', 'utf-8')
            msg.attach(text_plain)
            smtp = smtplib.SMTP()
            smtp.connect('smtp.qq.com')
            # 我们用set_debuglevel(1)就可以打印出和SMTP服务器交互的所有信息。
            # smtp.set_debuglevel(1)
            smtp.login(username, password)
            smtp.sendmail(sender, receiver, msg.as_string())
            smtp.quit()
            k_fold(5, train_list, device_ids)
    time.sleep(10)
