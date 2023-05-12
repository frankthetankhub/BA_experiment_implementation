import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import re
import json
import pandas as pd
import data_wrangling as dw
import itertools

omni_dict_location = "/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/omni_dict.json"

def plot(y_values):
    i=0
    metrics = ["Training Loss","Testing Loss","Training Accuracy","Testing Accuracy","Training Time per Epoch"]
    for metric in y_values:
        
        l=metric.shape[0]
        print(l)
        a = np.arange(l)
        plt.plot(a, metric, c="blue", label="Winning tickets") 
        # plt.plot(a, c, c="red", label="Random reinit") 
        t = metrics[i]
        print(t)
        plt.title(t)
        # plt.xlabel("Weights %") 
        # plt.ylabel("Test accuracy") 
        plt.xticks(np.arange(l,step=10), rotation ="vertical") 
        # plt.ylim(0,100)
        # plt.legend() 
        plt.grid(color="gray") 

        plt.savefig(f"{os.getcwd()}/plots/{i}.png", dpi=1200, bbox_inches='tight') 
        #plt.show()
        plt.close()
        i+=1


def load_raw_dicts(base_dir="/media/jan/9A2CA6762CA64CD7/ba_results/"):
    with open((base_dir+"large_scale_raw.json"), 'r') as f:
        large_scale_raw = json.load(f)
    with open((base_dir+"lth_raw.json"), 'r') as f:
        lth_raw = json.load(f)    
    return large_scale_raw, lth_raw

def plot_dataset_performance_averaged_set():
    df_set, df_lth_best, df_lth_raw = dw.load_dataframes()
    averaged = dw.make_avg_df(df_set)
    g = averaged[averaged.workers.eq(0)].groupby("dataset")["zeta_anneal","accuracy_test"] #  & averaged.zeta_anneal.eq(False)
    labels = []
    for label, data in g:
        print(data)
        labels.append(label+"anneal zeta")
        plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.zeta_anneal.eq(True)]["accuracy_test"]))))
        labels.append(label)
        plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.zeta_anneal.eq(False)]["accuracy_test"]))))
    plt.legend(labels)
    plt.savefig("plots/dataset_performance_averaged_anneal_comparison.png")

def plot_dataset_performance_set():
    df_set, df_lth_best, df_lth_raw = dw.load_dataframes()
    averaged = dw.make_avg_df(df_set)
    g = averaged[averaged.workers.eq(0)].groupby("dataset")["zeta_anneal","accuracy_test"] #  & averaged.zeta_anneal.eq(False)
    labels = []
    for label, data in g:
        print(data)
        labels.append(label+"anneal zeta")
        plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.zeta_anneal.eq(True)]["accuracy_test"]))))
        labels.append(label)
        plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.zeta_anneal.eq(False)]["accuracy_test"]))))
    plt.legend(labels)
    plt.savefig("plots/dataset_performance_anneal_comparison.png")




if __name__ == "__main__":
    plot_dataset_performance_set

    # df_set, df_lth_best, df_lth_raw = dw.load_dataframes()
    # averaged = dw.make_avg_df(df_set)
    # #averaged.plot()#(y="accuracy_train")
    # print(averaged.iloc[0]["accuracy_train"])
    # # groups = averaged.groupby(["start_imp","epsilon"])#,"workers"
    # # for name,data in groups:
    # #     print(data["accuracy_test"].shape)
    # #     print(data["accuracy_test"])
    # #     print(data[data.dataset.eq("cifar10")])
    # #     break

    # # exit()
    # # val1 = averaged[averaged.start_imp.eq(140) & averaged.epsilon.eq(20) & averaged.zeta_anneal.eq(False) & averaged.dataset.eq("cifar10") & averaged.arch_size.eq("small") & averaged.workers.eq(0)]
    # # val2 = averaged[averaged.start_imp.eq(140) & averaged.epsilon.eq(20) & averaged.zeta_anneal.eq(True) & averaged.dataset.eq("cifar10") & averaged.arch_size.eq("small") & averaged.workers.eq(0)]
    # # print(val1)
    # # print(val1.shape)
    # # print(val2)
    # # print(val2.shape)
    # # #print(np.array(groups["accuracy_test"]))
    # # plt.plot(np.arange(250), np.squeeze(val1["accuracy_train"].tolist()))
    # # plt.plot(np.arange(250), np.squeeze(val2["accuracy_train"].tolist()))
    # # plt.savefig("plots/test.png")
    # g = averaged[averaged.workers.eq(0)].groupby("dataset")["zeta_anneal","accuracy_test"] #  & averaged.zeta_anneal.eq(False)
    # #g = averaged[averaged.workers.eq(0)].groupby("zeta_anneal")["dataset","accuracy_test"] #  & averaged.zeta_anneal.eq(False)
    # print(g)
    # #print(g["accuracy_test"])
    # labels = []
    # for label, data in g:
    #     # print(label)
    #     print(data)
    #     #print(data.shape)
    #     # #data.accuracy_test.plot(kind="kde")
    #     # for p in data.accuracy_test:
    #     #     if p.shape[0] ==251:
    #     #         p = p[:250]
    #     # #print(data.accuracy_test)
    #     #     plt.plot(np.arange(250), p)
    #     labels.append(label+"anneal zeta")
    #     plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.zeta_anneal.eq(True)]["accuracy_test"]))))
    #     labels.append(label)
    #     plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.zeta_anneal.eq(False)]["accuracy_test"]))))
    #     # labels.append(str(label)+"anneal zeta")
    #     # plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.dataset.eq("cifar10")]["accuracy_test"]))))
    #     # labels.append(label)
    #     # plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.dataset.eq("mnist")]["accuracy_test"]))))
    #     # labels.append(label)
    #     # plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.dataset.eq("fashionmnist")]["accuracy_test"]))))
    # plt.legend(labels)
    # plt.savefig("plots/test2.png")