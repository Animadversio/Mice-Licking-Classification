import os
from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import torch
from neural_data_loading import load_data, rootdir, dataroot # , datadir
import time

def moving_average_nd(arr, window_size, axis=0):
    """
    Compute the moving average over a 2D array along the specified axis.

    Parameters:
    arr (numpy.ndarray): The input 2D array.
    window_size (int): The size of the moving average window.
    axis (int): The axis along which to compute the moving average.

    # Example usage
        avg_n = 2
        mva_neural_mat = moving_average_nd(neural_mat, window_size=avg_n, axis=-1)
        print (mva_neural_mat.shape)
        mva_neural_mat_subsp = mva_neural_mat[:, :, ::avg_n]
        print(mva_neural_mat_subsp.shape)

    Returns:
    numpy.ndarray: The array after applying the moving average.
    """
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=axis, arr=arr)


def main_sweep(args):
    from neural_data_loading import dataroot
    datadir = join(dataroot, args.datadir)
    outdir = args.outdir 
    dataset_label = args.dataset_label
    only_firstbout = args.only_firstbout
    model_str_list = args.model_str
    avg_n_list = args.avg_n
    seed = args.seed
    test_size = args.test_size
    if not dataset_label:
        # Extract the final part of datadir
        dataset_label = os.path.basename(os.path.normpath(datadir))
    print("data path", datadir)
    print("use dataset_label:", dataset_label)
        
    if outdir == "":
        outdir = join(rootdir, "Figures", dataset_label) 
        #"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/SabatiniShijiaLickingClassifier/Figures"
    os.makedirs(outdir, exist_ok=True)
    
    neural_mat, lastlicksId_mat, firstboutId_mat = load_data(datadir)
    for only_firstbout in [False, True]:
        if only_firstbout:
            lick_mat = lastlicksId_mat & firstboutId_mat
        else:
            lick_mat = lastlicksId_mat
        # fit model no training data
        for model_str in model_str_list: # "gblinear"
            df_col = []
            # todo, fix this, only test one setting. 
            print("using first bount lick", only_firstbout)
            for avg_n in avg_n_list:
                print("Average bin num:", avg_n) 
                # average firing rate in bins together
                mva_neural_mat = moving_average_nd(neural_mat, window_size=avg_n, axis=-1)
                print("Convolved tensor shape", mva_neural_mat.shape)
                mva_neural_mat_subsp = mva_neural_mat[:, :, ::avg_n]
                print("Convolved & subsampled tensor shape", mva_neural_mat_subsp.shape)
                # normalize
                mva_neuron_mean = mva_neural_mat_subsp.mean(axis=(0, 2))
                mva_neuron_std = mva_neural_mat_subsp.std(axis=(0, 2))
                mva_neural_tsr_zscore = ((mva_neural_mat_subsp.astype("float32") - mva_neural_mat_subsp.mean(axis=(0, 2), keepdims=True))
                                    / mva_neural_mat_subsp.std(axis=(0, 2), keepdims=True))
                print("Normalize each channel across time & trial done.")
                total_bin_num = mva_neural_mat_subsp.shape[-1]
                for lag in range(1, total_bin_num+1):
                    # for lag in [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]:
                    for init_bin in range(0, total_bin_num, lag):
                        # select the training features via the time slices.
                        neural_mat_zscore = mva_neural_tsr_zscore[:, :, init_bin:init_bin+lag]
                        neural_mat_zscore = neural_mat_zscore.reshape(neural_mat_zscore.shape[0], -1)
                        bin_num = mva_neural_tsr_zscore[:, :, init_bin:init_bin+lag].shape[-1]
                        if bin_num == 0:
                            break
                        data_str = f"{5 * avg_n}ms_from_{init_bin}_{bin_num}bins"
                        y_mat = lick_mat[0]
                        # split data into train and test sets
                        X_train, X_test, y_train, y_test = train_test_split(neural_mat_zscore,
                                                y_mat, test_size=test_size, random_state=seed)
                        t0 = time.time()
                        if model_str == "logregress":
                            model = LogisticRegressionCV(penalty="l2", cv=5, 
                                                         random_state=seed, max_iter=500,
                                                         class_weight="balanced")
                        else:
                            model = XGBClassifier(booster=model_str,
                                    objective="binary:logistic",
                                    eval_metric="logloss",
                                    use_label_encoder=False,
                                    device="cuda", )
                        model.fit(X_train, y_train)
                        # make predictions for test data
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)
                        t1 = time.time()
                        # evaluate predictions
                        acc_train = accuracy_score(y_train, y_train_pred)
                        acc_test  = accuracy_score(y_test, y_test_pred)
                        cm_train = confusion_matrix(y_train, y_train_pred)
                        cm_test  = confusion_matrix(y_test, y_test_pred)
                        print(f"Avg bins {avg_n}, from {init_bin} to {init_bin + bin_num} bin")
                        print(f"Accuracy: train: {acc_train:.3f} test: {acc_test:.3f}")
                        print(f"TP / (TP + FP): train: {cm_train[1, 1] / cm_train[:, 1].sum():.3f} "
                            f"test: {cm_test[1, 1] / cm_test[:, 1].sum():.3f} ")
                        print(f"TP / (TP + FN): train: {cm_train[1, 1] / cm_train[:, 1].sum():.3f} "
                            f"test: {cm_test[1, 1] / cm_test[1, :].sum():.3f} ")
                        print("train_confmat\n", cm_train)
                        print("test_confmat\n", cm_test)
                        print (f"Time for model: {t1-t0:.2f}s")
                        df_col.append({"init_bin": init_bin, "lag": lag, "bin_num": bin_num, "avg_bin_num": avg_n, 
                                    "acc_train": acc_train, "acc_test": acc_test,
                                    "tp_train": cm_train[1, 1] / cm_train[:, 1].sum(),
                                    "tp_test": cm_test[1, 1] / cm_test[:, 1].sum(),
                                    "cm_train": cm_train, "cm_test": cm_test,
                                    "model_str": model_str, "data_str": data_str,
                                    "only_firstbout_y": only_firstbout})
                        if bin_num < lag:
                            break
                        
            df = pd.DataFrame(df_col)
            df.to_csv(join(outdir, f"confmat_{dataset_label}_{model_str}_avg_multi_ms_tab_{'firstbouty' if only_firstbout else 'allbouty'}.csv"), index=False)
        
    # merge all methods into a master table
    df_all = []
    for only_firstbout in [False, True]:
        for model_str in model_str_list:
            df = pd.read_csv(join(outdir, f"confmat_{dataset_label}_{model_str}_avg_multi_ms_tab_{'firstbouty' if only_firstbout else 'allbouty'}.csv"))
            df_all.append(df)
        
    df_all = pd.concat(df_all, axis=0)
    df_all["time_beg"] = df_all["init_bin"] * 5 * df_all["avg_bin_num"]
    df_all["time_end"] = (df_all["init_bin"] + df_all["bin_num"]) * 5 * df_all["avg_bin_num"]
    df_all.to_csv(join(outdir, f"confmat_{dataset_label}_allmethod_avg_multi_ms_tab.csv"), index=False)
    return df_all


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run neural data analysis.")
    
    parser.add_argument("--datadir", type=str, default="", help="Directory that has paired neural recording and behavior.")
    parser.add_argument("--dataset_label", type=str, default="", help="Label for the dataset.")
    parser.add_argument("--outdir", type=str, default="", help="Directory that has paired neural recording and behavior.")
    parser.add_argument("--model_str", type=str, default=["gblinear", "gbtree", "dart", "logregress"], nargs="+", help="name of models used")
    parser.add_argument("--avg_n", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 40], nargs="+", help="bin average int to use")
    parser.add_argument("--test_size", type=float, default=0.2, help="Size of the test dataset as a fraction.")
    parser.add_argument("--seed", type=int, default=41, help="Seed for random number generation.")
    parser.add_argument("--only_firstbout", type=bool, default=False, help="Flag to use only firstbout data.")
    
    args = parser.parse_args()
    main_sweep(args)

# neuron_mean = neural_mat.mean(axis=(0, 2))
# neuron_std = neural_mat.std(axis=(0, 2))
# # normalize
# neural_tsr_zscore = ((neural_mat.astype("float32") - neural_mat.mean(axis=(0, 2), keepdims=True))
#                      / neural_mat.std(axis=(0, 2), keepdims=True))
# device = "cpu"
# neural_tsr_zscore_th = torch.from_numpy(neural_tsr_zscore).float().to(device)
# lick_mat_th = torch.from_numpy(lick_mat[0]).long().to(device)


# neural_mat_zscore = neural_tsr_zscore.reshape(neural_tsr_zscore.shape[0], -1)
# y_mat = lick_mat[0]
# X_train, X_test, y_train, y_test = train_test_split(neural_mat_zscore,
#                         y_mat, test_size=test_size, random_state=seed)

# plot confusion matrix
# figh, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["no lick", "lick"])
# disp.plot(ax=axs[0])
# axs[0].set_title("Train")
# disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["no lick", "lick"])
# disp.plot(ax=axs[1])
# axs[1].set_title("Test") 
# plt.suptitle(f"Confusion matrix of XGBoost classifier\n Classifier: {model_str} | Data: {data_str}")
# plt.savefig(join(outdir,f"confmat_{model_str}_{data_str}.png"))
# plt.savefig(join(outdir,f"confmat_{model_str}_{data_str}.pdf"))
# plt.show()