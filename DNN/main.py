# %%
import numpy as np
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import time
import scipy.io as sio

from models.DNN import DNN
from utils.Function import *
from utils.Load_data import Load_data, Load_data_split
from utils.Utils import AverageMeter,goodness_of_fit

device= "cuda" if torch.cuda.is_available else "cpu" 
device = "cpu"
time_now = time.localtime()

# %%
input_size = 8
hidden_size1 = 80
hidden_size2 = 40
output_size = 3
n_epoch = 1000
batch_size = 16
pretrained = False
test = False
all_test = False
residual_save = False
model_path = r""
data_path = r"./DNN/yrprvcdata_sliced_all_log.mat"

# %%
Ctrain, Ctest, Rtrain, Rtest, fttrain, fttest = Load_data_split(data_path)
print(Ctrain.shape, Ctest.shape, Rtrain.shape, Rtest.shape, fttrain.shape, fttrain.shape)

input_train = torch.tensor(Ctrain, dtype=torch.float32)
input_val = torch.tensor(Ctest, dtype=torch.float32)
input_test = torch.tensor(Ctest, dtype=torch.float32)
# input_test = torch.tensor(Ctrain, dtype=torch.float32)
input = torch.concat([input_train, input_test])

output_train = torch.tensor(Rtrain, dtype=torch.float32)
output_val = torch.tensor(Rtest, dtype=torch.float32)
output_test = torch.tensor(Rtest, dtype=torch.float32)
# output_test = torch.tensor(Rtrain, dtype=torch.float32)
output = torch.concat([output_train, output_test])

c_train = torch.tensor(fttrain, dtype=torch.float32)
c_val = torch.tensor(fttest, dtype=torch.float32)
c_test = torch.tensor(fttest, dtype=torch.float32)
# c_test = torch.tensor(fttrain, dtype=torch.float32)
c = torch.concat([c_train, c_test])


print(input_train.shape, output_train.shape, c_train.shape)
print(input_val.shape, output_val.shape, c_val.shape)
print(input_test.shape, output_test.shape, c_test.shape)
print(input.shape, output.shape, c.shape)

# %%
train_dataset = torch.utils.data.TensorDataset(input_train, output_train, c_train)
val_dataset = torch.utils.data.TensorDataset(input_val, output_val, c_val)
test_dataset = torch.utils.data.TensorDataset(input_test, output_test, c_test)
dataset = torch.utils.data.TensorDataset(input, output, c)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# %%
model = DNN(input_size, hidden_size1, hidden_size2, output_size).to(device) #
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001) 
loss_func = nn.MSELoss()

for p in model.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)
    else:
        torch.nn.init.uniform_(p)

if pretrained:
    model.load_state_dict(torch.load(model_path))

# %%
if not test:
    train_loss_show = AverageMeter()
    print("=" * 100)
    for i in range(n_epoch):
        model.train()
        loop_train = tqdm(train_dataloader, total = len(train_dataloader))
        for batch_data, batch_label, c_train in loop_train:
            # batch_data = batch_data.cuda() 
            # batch_label = batch_label.cuda() 
            theta = model(batch_data)
            optimizer.zero_grad()
            predict_train = theta[:, 0].unsqueeze(1) * c_train[:, 0].unsqueeze(1) + theta[:, 1].unsqueeze(1) * c_train[:, 1].unsqueeze(1) + theta[:, 2].unsqueeze(1)
            loss = loss_func(predict_train, batch_label)
            loss.requires_grad_(True)
            loss.backward() 
            optimizer.step()
            train_loss_show.update(loss.data, 1)
            loop_train.set_description(f'Train Epoch [{"%04d"}/{"%04d"}]' % (i+1, n_epoch))
            loop_train.set_postfix({"train_loss":train_loss_show.average.item()})

        if (i+1)%20 == 0:
            print("=" * 100)
            print("Model Validation:")
            val_loss_show = AverageMeter()
            loop_val = tqdm(val_dataloader, total = len(val_dataloader))
            for batch_data, batch_label, c_val in loop_val:
                theta = model(batch_data)
                predict_val = theta[:, 0].unsqueeze(1) * c_val[:, 0].unsqueeze(1) + theta[:, 1].unsqueeze(1) * c_val[:, 1].unsqueeze(1) + theta[:, 2].unsqueeze(1)
                loss = loss_func(predict_val, batch_label)
                val_loss_show.update(loss.data, 1)
                loop_val.set_description(f'Val Epoch')
                loop_val.set_postfix({"val_loss":val_loss_show.average.item()})
            print("=" * 100)

# %%
test_loss_show = AverageMeter()
prediction = torch.tensor([])
label = torch.tensor([])
if not all_test:
    loop_test = tqdm(test_dataloader, total = len(test_dataloader))
else:
    loop_test = tqdm(dataloader, total = len(dataloader))
for batch_data, batch_label, c_test in loop_test:
        theta = model(batch_data)
        predict_test = theta[:, 0].unsqueeze(1) * c_test[:, 0].unsqueeze(1) + theta[:, 1].unsqueeze(1) * c_test[:, 1].unsqueeze(1) + theta[:, 2].unsqueeze(1)
        prediction = torch.concat([prediction, predict_test])
        label = torch.concat([label, batch_label])
        loss = loss_func(predict_test, batch_label)
        test_loss_show.update(loss.data, 1)
        loop_test.set_description(f'Test')
        loop_test.set_postfix({"test_loss":test_loss_show.average.item()})
print("=" * 100)

# %%
prediction = torch.flatten(prediction).detach().numpy()
label = torch.flatten(label).detach().numpy()
R_square = goodness_of_fit(prediction, label)
print("R^2 = ", R_square)
print("=" * 100)

if not test:
    torch.save(model.state_dict(), r"./DNN/logs/model_state_dict_" + data_path.split("/")[1] + "_" + str(n_epoch) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + ".pth")
    torch.save(model, r"./DNN/logs/model_" + data_path.split("/")[1] + "_" + str(n_epoch) + "_" +  time.strftime("%Y-%m-%d-%H-%M-%S", time_now) + ".pth")

# %%
if residual_save:
    result_mat = "./results/result_prediction_label.mat"
    residual = prediction - label
    print(max(residual), min(residual), np.mean(residual), np.std(residual))
    sio.savemat(result_mat, {'prediction': prediction, 'label': label, 'residual': residual})
