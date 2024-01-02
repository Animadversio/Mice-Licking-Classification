# read in mat file
import mat73
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import scipy.io as sio
rootdir = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/SabatiniShijiaLickingClassifier"
neural_mat = sio.loadmat(join(rootdir, "allLicksNeural 1.mat"))
lick_mat = sio.loadmat(join(rootdir, "lastLickIdentity 1.mat"))
neural_mat = neural_mat['allLicksNeural']
lick_mat = lick_mat['lastLickIdentity']
#%%
# split the data into train and test
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader
#%%
neural_mat = neural_mat
neuron_mean = neural_mat.mean(axis=(0,2))
neuron_std = neural_mat.std(axis=(0,2))
# normalize
neural_mat_zscore = ((neural_mat.astype("float32") - neural_mat.mean(axis=(0, 2), keepdims=True))
              / neural_mat.std(axis=(0, 2), keepdims=True))
device = "cpu"
neural_mat_zscore_th = torch.from_numpy(neural_mat_zscore).float().to(device)
lick_mat_th = torch.from_numpy(lick_mat[0]).long().to(device)
#%%
# split the data into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    neural_mat_zscore_th, lick_mat_th,
    test_size=0.2, random_state=41, shuffle=True,)
# TODO: use the label to divide
# convert to torch tensor
# Xtrain = torch.from_numpy(Xtrain, ).float().mps()
# Xtest = torch.from_numpy(Xtest, ).float().mps()
# ytrain = torch.from_numpy(ytrain, ).long().mps()
# ytest = torch.from_numpy(ytest, ).long().mps()
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        layers += [nn.AdaptiveMaxPool1d(1)]
        layers += [nn.Flatten()]
        layers += [nn.Linear(num_channels[-1], 2)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

#%%
class_freq = ytrain.bincount().float()
class_weight = class_freq.sum() / class_freq
# define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weight)
#%%
# https://youtrack.jetbrains.com/issue/PY-61385/PyCharm-Python-Console-crashes-everytime-I-use-PyTorch
# train the network
num_channel = neural_mat.shape[1]
num_hidden = 128
net = TemporalConvNet(num_channel, [num_hidden]*4, kernel_size=3, dropout=0.2)
optimizer = optim.Adam(net.parameters(), lr=0.005)
batchsize = 512
num_epochs = 50
train_loss = []
test_loss = []
train_acc = []
test_acc = []
train_tpallp_ratio = []
test_tpallp_ratio = []
train_detect_ratio = []
test_detect_ratio = []
train_dataset = TensorDataset(Xtrain, ytrain)
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
for epoch in range(num_epochs):
    net.train()
    for Xbatch, ybatch in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(Xbatch)
        loss = criterion(outputs, ybatch)
        loss.backward()
        optimizer.step()
    # forward + backward + optimize
    with torch.no_grad():
        outputs = net(Xtrain)
    loss = criterion(outputs, ytrain)
    pred = outputs.argmax(dim=1).detach().cpu()
    confmat_train = confusion_matrix(ytrain.cpu(), pred)
    train_loss.append(loss.item())
    train_acc.append(accuracy_score(ytrain, pred))
    train_tpallp_ratio.append(confmat_train[1, 1] / confmat_train[:, 1].sum())
    train_detect_ratio.append(confmat_train[1, 1] / confmat_train[1, :].sum())
    # evaluate on the test set
    net.eval()
    with torch.no_grad():
        outputs = net(Xtest)
    loss = criterion(outputs, ytest)
    pred = outputs.argmax(dim=1).detach().cpu()
    confmat_test = confusion_matrix(ytest.cpu(), pred)
    # print statistics
    test_loss.append(loss.item())
    test_acc.append(accuracy_score(ytest, pred))
    test_tpallp_ratio.append(confmat_test[1, 1] / confmat_test[:, 1].sum())
    test_detect_ratio.append(confmat_test[1, 1] / confmat_test[1, :].sum())
    if epoch % 1 == 0:
        print(f"Epoch {epoch} | Loss: Train {train_loss[-1]:.3f} Test {test_loss[-1]:.3f} | "
              f"Acc: Train {train_acc[-1]:.3f} Test {test_acc[-1]:.3f} | "
              f"TP/(TP+FP): Train {train_tpallp_ratio[-1]:.3f} Test {test_tpallp_ratio[-1]:.3f} | "
              f"TP/(TP+miss): Train {train_detect_ratio[-1]:.3f} Test {test_detect_ratio[-1]:.3f}")

#%%
# plot confusion matrix for the test set and train set
# confmat = confusion_matrix(ytest, outputs.argmax(dim=1))
train_outputs = net(Xtrain)
train_confmat = confusion_matrix(ytrain, train_outputs.argmax(dim=1))
test_outputs = net(Xtest)
test_confmat = confusion_matrix(ytest, test_outputs.argmax(dim=1))
# print the confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ConfusionMatrixDisplay(train_confmat).plot(ax=ax[0])
ax[0].set_title("Train set")
ConfusionMatrixDisplay(test_confmat).plot(ax=ax[1])
ax[1].set_title("Test set")
plt.show()
#%%
print(train_confmat)
print(test_confmat)



#%%
#%%
num_channel = neural_mat.shape[1]
num_hidden = 256
num_output = 2
# Define a temporal convolutional network with 1D convolutions
net = nn.Sequential(
    nn.Conv1d(num_channel, num_hidden, kernel_size=3, padding=1),
    nn.BatchNorm1d(num_hidden),
    nn.ReLU(),
    nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1),
    nn.BatchNorm1d(num_hidden),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1),
    nn.BatchNorm1d(num_hidden),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1),
    nn.BatchNorm1d(num_hidden),
    nn.ReLU(),
    # nn.Conv1d(num_hidden, num_hidden, kernel_size=3, padding=1),
    # nn.BatchNorm1d(num_hidden),
    # nn.ReLU(),
    # nn.Dropout(0.5),
    # pooling
    nn.AdaptiveMaxPool1d(1),
    nn.Flatten(),
    nn.Linear(num_hidden, num_output),
    # nn.Softmax(dim=1)
)
#%%
# define the loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.7, 4.]))
optimizer = optim.Adam(net.parameters(), lr=0.005)
#%%
# https://youtrack.jetbrains.com/issue/PY-61385/PyCharm-Python-Console-crashes-everytime-I-use-PyTorch
# train the network
num_epochs = 20
train_loss = []
test_loss = []
train_acc = []
test_acc = []
train_tpallp_ratio = []
test_tpallp_ratio = []
net.to(device)
# Xtrain = Xtrain.to(device)
# Xtest = Xtest.to(device)
# ytrain = ytrain.to(device)
# ytest = ytest.to(device)
for epoch in range(num_epochs):
    net.train()
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = net(Xtrain)
    loss = criterion(outputs, ytrain)
    loss.backward()
    optimizer.step()
    pred = outputs.argmax(dim=1).detach().cpu()
    confmat_train = confusion_matrix(ytrain.cpu(), pred)
    train_loss.append(loss.item())
    train_acc.append(accuracy_score(ytrain, pred))
    train_tpallp_ratio.append(confmat_train[1, 1] / confmat_train[:, 1].sum())
    # evaluate on the test set
    net.eval()
    with torch.no_grad():
        outputs = net(Xtest)
    loss = criterion(outputs, ytest)
    pred = outputs.argmax(dim=1).detach().cpu()
    confmat_test = confusion_matrix(ytest.cpu(), pred)
    # print statistics
    test_loss.append(loss.item())
    test_acc.append(accuracy_score(ytest, pred))
    test_tpallp_ratio.append(confmat_test[1, 1] / confmat_test[:, 1].sum())
    if epoch % 1 == 0:
        print(f"Epoch {epoch} | Loss: Train {train_loss[-1]:.3f} Test {test_loss[-1]:.3f} | "
              f"Acc: Train {train_acc[-1]:.3f} Test {test_acc[-1]:.3f} | "
              f"TP/(TP+FP): Train {train_tpallp_ratio[-1]:.3f} Test {test_tpallp_ratio[-1]:.3f}")
#%%
# plot confusion matrix for the test set and train set
# confmat = confusion_matrix(ytest, outputs.argmax(dim=1))
train_outputs = net(Xtrain)
train_confmat = confusion_matrix(ytrain, train_outputs.argmax(dim=1))
test_outputs = net(Xtest)
test_confmat = confusion_matrix(ytest, test_outputs.argmax(dim=1))
#%%
# print the confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ConfusionMatrixDisplay(train_confmat).plot(ax=ax[0])
ax[0].set_title("Train set")
ConfusionMatrixDisplay(test_confmat).plot(ax=ax[1])
ax[1].set_title("Test set")
plt.show()
#%%
# plot roc curve
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
#%%
test_prob = F.softmax(test_outputs.detach(), dim=1)
train_prob = F.softmax(train_outputs.detach(), dim=1)
# plot the roc curve for the test set and train set
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fpr, tpr, thresh = roc_curve(ytest, test_prob[:, 1])
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax[0])
ax[0].set_title("Test set")
fpr, tpr, thresh = roc_curve(ytrain, train_prob[:, 1])
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax[1])
ax[1].set_title("Train set")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fpr, tpr, thresh = roc_curve(ytest, test_prob[:, 1])
ax[0].plot(thresh, fpr, label="fpr")
ax[0].plot(thresh, tpr, label="tpr")
ax[0].set_title("Test set")
fpr, tpr, thresh = roc_curve(ytrain, train_prob[:, 1])
ax[1].plot(thresh, fpr, label="fpr")
ax[1].plot(thresh, tpr, label="tpr")
ax[1].set_title("Train set")
plt.legend()
plt.show()
