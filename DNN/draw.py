# %%
import numpy as np
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import time
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from models.DNN import DNN
from utils.Function import *
from utils.Load_data import Load_data, Load_data_split
from utils.Utils import AverageMeter,goodness_of_fit

device= "cuda" if torch.cuda.is_available else "cpu" 
device = "cpu"
time_now = time.localtime()
input_size = 8
hidden_size1 = 80
hidden_size2 = 40
output_size = 3
batch_size = 16
model_path = r"logs\model_state_dict_yrprvcdata_sliced_all.mat_1000_2023-05-24-20-47-19.pth"
variable_name = ["商品价格指数", "社会消费品零售总额", "乡村恩格尔系数", "产业结构指数", "发电量", "城镇化率", "城镇恩格尔系数", "财政支出"]

# %%
Ctrain, Ctest, Rtrain, Rtest, fttrain, fttest = Load_data_split(r".\yrprvcdata_sliced_all.mat")
print(Ctrain.shape, Ctest.shape, Rtrain.shape, Rtest.shape, fttrain.shape, fttrain.shape)

input_train = torch.tensor(Ctrain, dtype=torch.float32)
input_val = torch.tensor(Ctest, dtype=torch.float32)
input_test = torch.tensor(Ctest, dtype=torch.float32)
input = torch.concat([input_train, input_test])

output_train = torch.tensor(Rtrain, dtype=torch.float32)
output_val = torch.tensor(Rtest, dtype=torch.float32)
output_test = torch.tensor(Rtest, dtype=torch.float32)
output = torch.concat([output_train, output_test])

c_train = torch.tensor(fttrain, dtype=torch.float32)
c_val = torch.tensor(fttest, dtype=torch.float32)
c = torch.tensor(fttest, dtype=torch.float32)
c = torch.concat([c_train, c])

print(input_train.shape, output_train.shape, c_train.shape)
print(input_val.shape, output_val.shape, c_val.shape)
print(input_test.shape, output_test.shape, c.shape)
print(input.shape, output.shape, c.shape)

# %%
input_max = [max(x).item() for x in torch.transpose(input, 1, 0)]
print(input_max)
input_min = [min(x).item() for x in torch.transpose(input, 1, 0)]
print(input_min)

# %%
num_step = 10000
dataset_input = np.zeros([input.shape[1], num_step])
for i in range(input.shape[1]):
    dataset_input[i] = np.linspace(input_min[i], input_max[i], num_step)
print(dataset_input.shape)

# %%
dataset_zero = np.zeros([1, num_step])
dataset_pre = np.zeros([input.shape[1], input.shape[1], num_step])
for i in range(input.shape[1]):
    for j in range(input.shape[1]):
        if i==j:
            dataset_pre[i,j,:] = dataset_input[i,:]
        else:
            dataset_pre[i,j,:] = dataset_zero
print(dataset_pre.shape)
print(dataset_pre[2,2,:].shape)
print(dataset_pre[2,2,:])
dataset_pre = torch.tensor(dataset_pre, dtype=torch.float32)

# %%
print(c.shape)
c_mean = [torch.mean(x).item() for x in torch.transpose(c, 1, 0)]
print(c_mean)
c = np.ones([num_step, c.shape[1]])
c[:,0] = c[:,0] * c_mean[0]
c[:,1] = c[:,1] * c_mean[1]
print(c[:,0])
print(c[:,1])
c = torch.tensor(c, dtype=torch.float32)
print(c.shape)

# %%
model = DNN(input_size, hidden_size1, hidden_size2, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001) 
loss_func = nn.MSELoss()
model.load_state_dict(torch.load(model_path))

# %%
prediction = torch.tensor([])
theta = torch.tensor([])

# %%
print(c.shape)
for i in range(input.shape[1]):
    print(c.shape)
    dataset = torch.utils.data.TensorDataset(torch.transpose(dataset_pre[i], 1, 0), c)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    prediction_temp = torch.tensor([])
    theta_temp = torch.tensor([])
    loss_show = AverageMeter()
    loop = tqdm(dataloader, total = len(dataloader))
    for batch_data, batch_c in loop:
            batch_theta = model(batch_data)
            batch_prediction = batch_theta[:, 0].unsqueeze(1) * batch_c[:, 0].unsqueeze(1) + batch_theta[:, 1].unsqueeze(1) * batch_c[:, 1].unsqueeze(1) + batch_theta[:, 2].unsqueeze(1)
            prediction_temp = torch.concat([prediction_temp, batch_prediction])
            theta_temp = torch.concat([theta_temp, batch_theta])
    print("=" * 100)

    prediction = torch.concat([prediction, prediction_temp], dim=1)
    theta = torch.concat([theta, theta_temp.unsqueeze(2)], dim=2)

# %%
prediction = prediction.detach().numpy()
theta = theta.detach().numpy()
print(prediction.shape)
print(theta.shape)
plt.rcParams['font.sans-serif'] = "Microsoft YaHei"

# %%
config = {# 图表绘制初始字体字典设置
    "font.family": "serif",  # 使用衬线体
    "font.serif": ["Microsoft YaHei"],  # 全局默认使用衬线宋体
    "font.size": 12,
    "font.weight": "bold",
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
}
plt.rcParams.update(config)

ticklabels_style = {# 刻度绘制初始字体字典设置
    "fontname": "Arial",
    "fontsize": 10,
}

# %%
print("gdp")
for i in range(input.shape[1]):
    if i==7:
        print(i+1)
        figure, ax1 = plt.subplots()
        ax1.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
        plt.plot(dataset_input[i,:], prediction[:,i], '--', c="red", linewidth=2)
        ax1.set_xlim([-1, 8+0.001])
        ax1.set_xticks(np.arange(-1, 8+0.001, 1))
        ax1.set_ylim(bottom=8, top=12.2+0.001)
        ax1.set_yticks(np.arange(8, 12+0.001, 0.5))
        ax1.set_xlabel(variable_name[i]+"(标准化)", weight='bold', fontsize=12)# x轴标题
        ax1.set_ylabel('GDP(标准化)', weight='bold', fontsize=12)# y轴标题
        plt.gca().spines['left'].set_linewidth(1.2)# 左边框宽度
        plt.gca().spines['bottom'].set_linewidth(1.2)# 下边框宽度
        plt.gca().spines['right'].set_linewidth(1.2)# 右边框宽度
        plt.gca().spines['top'].set_linewidth(1.2)# 上边框宽度
        plt.savefig(".\images\第"+str(i+1)+"个工具变量"+variable_name[i]+"-GDP.jpg", dpi=300)
        plt.show()
        break

# %%
print("theta")
for i in range(input.shape[1]):
    if i==7:
        print(i+1)
        figure, ax2 = plt.subplots()
        ax2.grid(axis='y', linestyle='-.', alpha=0.1)# 背景网格
        plt.plot(dataset_input[i,:], theta[:,0,i], ':', label=r"$\alpha$", linewidth=2, c="#E6AF0C")
        plt.plot(dataset_input[i,:], theta[:,1,i], '--', label=r"$\beta$", linewidth=2, c="#78A51E")
        plt.plot(dataset_input[i,:], theta[:,2,i], '-.',  label=r"$A$", linewidth=2, c="#E52287")
        ax2.set_xlim([-1, 8+0.001])
        ax2.set_xticks(np.arange(-1, 8+0.001, 0.5))
        ax2.set_ylim(bottom=0, top=4.5+0.001)
        ax2.set_yticks(np.arange(0, 4.5+0.001, 0.5))
        ax2.set_xlabel(variable_name[i]+"(标准化)", weight='bold', fontsize=12)# x轴标题
        ax2.set_ylabel('系数值', weight='bold', fontsize=12)# y轴标题
        plt.gca().spines['left'].set_linewidth(1.2)# 左边框宽度
        plt.gca().spines['bottom'].set_linewidth(1.2)# 下边框宽度
        plt.gca().spines['right'].set_linewidth(1.2)# 右边框宽度
        plt.gca().spines['top'].set_linewidth(1.2)# 上边框宽度
        ax2.legend(fontsize=12, frameon=False, ncol=3, loc='upper right')# 图例
        plt.savefig("./images/第"+str(i+1)+"个工具变量"+variable_name[i]+"-theta.jpg", dpi=300)
        plt.show()