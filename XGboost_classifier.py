# read in mat file
import mat73
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import scipy.io as sio
# scipy.io.loadmat()
rootdir = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/SabatiniShijiaLickingClassifier"
neural_mat = sio.loadmat(join(rootdir, "allLicksNeural 1.mat"))
lick_mat = sio.loadmat(join(rootdir, "lastLickIdentity 1.mat"))
neural_mat = neural_mat['allLicksNeural']
lick_mat = lick_mat['lastLickIdentity']
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
# use xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#%%
# split data into train and test sets
seed = 41
test_size = 0.2
neural_mat_zscore = neural_mat_zscore.reshape(neural_mat_zscore.shape[0], -1)
X_train, X_test, y_train, y_test = train_test_split(neural_mat_zscore,
                                                    lick_mat[0], test_size=test_size, random_state=seed)

#%%
import matplotlib.pyplot as plt
# fit model no training data
model = XGBClassifier(booster="gblinear",
                      objective="binary:logistic",
                      eval_metric="logloss",
                      use_label_encoder=False,
                      device="cpu",)
model.fit(X_train, y_train)
#%%
# make predictions for test data
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)
# evaluate predictions
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_pred)
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_pred)
print(f"Accuracy: train: {acc_train:.3f} test: {acc_test:.3f}")
print(f"TP / (TP + FP): train: {cm_train[1, 1] / cm_train[:, 1].sum():.3f} "
    f"test: {cm_test[1, 1] / cm_test[:, 1].sum():.3f} ")
print(f"TP / (TP + FN): train: {cm_train[1, 1] / cm_train[:, 1].sum():.3f} "
    f"test: {cm_test[1, 1] / cm_test[1, :].sum():.3f} ")
print("train_confmat\n", cm_train)
print("test_confmat\n", cm_test)
#%%
figh, axs = plt.subplots(1, 2, figsize=(10, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["no lick", "lick"])
disp.plot(ax=axs[0])
axs[0].set_title("Train")
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["no lick", "lick"])
disp.plot(ax=axs[1])
axs[1].set_title("Test")
plt.show()
#%%

