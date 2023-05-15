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
    averaged = dw.load_averaged_dataframes()
    #averaged = dw.make_avg_df(df_set)
    g = averaged[averaged.workers.eq(0)].groupby(["dataset","arch_size"])["zeta_anneal","accuracy_test"] #  & averaged.zeta_anneal.eq(False)
    labels = []
    for label, data in g:
        print(label)
        print(data)
        labels.append(str(label)+"anneal zeta")
        d1 = np.array(data[data.zeta_anneal.eq(True)]["accuracy_test"].tolist())
        print(d1)
        d1 = d1.reshape(-1,250)
        d1 = np.squeeze(np.mean(d1, axis = 0))
        plt.plot(np.arange(250), d1)
        labels.append(str(label))
        plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.zeta_anneal.eq(False)]["accuracy_test"].tolist()), axis=0)))
    plt.legend(labels)
    plt.savefig("plots/dataset_performance_anneal_comparison.png")

def plot_individual_performance_config():
    averaged = dw.load_averaged_dataframes()
    for config in list(range(1,13)):#["cifar10", "mnist", "fashionmnist"]:
        print(averaged)
        # g = averaged[]
        # print(g)
        g = averaged[averaged.workers.eq(0) & averaged.config.eq(config)].groupby(["dataset", "arch_size"])["zeta_anneal","accuracy_test"] #
        labels = []
        
        plt.figure(figsize=(20,10))
        for label, data in g:
            plt.vlines(x=[140, 200], ymin=0, ymax = 1, colors="gray", ls="--")
            print(label)
            # print(type(label))
            labels.append(str(label)+"anneal zeta")
            d1 = data[data.zeta_anneal.eq(True)& data.config.eq(config)]["accuracy_test"]# 
            #print(d1)
            d1 = np.array(d1.tolist())
            d1 = d1.reshape(-1,250)
            d1 = np.squeeze(np.mean(d1, axis = 0))
            plt.plot(np.arange(250), d1)

            labels.append(str(label)+"no anneal")
            d1 = np.array(data[data.zeta_anneal.eq(False)& data.config.eq(config)]["accuracy_test"].tolist())
            d1 = d1.reshape(-1,250)
            d1 = np.squeeze(np.mean(d1, axis = 0))
            plt.plot(np.arange(250), d1,)
            plt.legend(labels)
        plt.savefig(f"plots/config_comparison/config{config}_performance")
        plt.close()

def plot_compare_all_configs():
    averaged,_ = dw.load_averaged_dataframes()
    
    g = averaged[averaged.workers.eq(0)].groupby(["config"])["zeta_anneal","accuracy_test"]
    labels = []
    
    plt.figure(figsize=(15,10))
    plt.vlines(x=[140, 200], ymin=0, ymax = 1, colors="gray", ls="--")
    for label, data in g:
        print(label)
        # print(type(label))
        labels.append(str(label))
        if label<5: ls_stil = "-"
        elif label >8: ls_stil = "--"
        else: ls_stil =":"
        print(data)
        d1 = data["accuracy_test"].tolist()
        # print([len(l) for l in d1])
        # d1 = np.array(np.array(d for d in d1))
        d1 = np.array(d1).reshape(-1,250)
        d1 = np.squeeze(np.mean(d1, axis = 0))
        plt.plot(np.arange(250), d1, ls=ls_stil)
        plt.legend(labels)
    plt.tight_layout()
    plt.savefig(f"plots/config_comparison/all_configs_averaged")
    plt.close()

def plot_compare_set_lth():
    df_set, df_lth = dw.load_averaged_dataframes()
    lth_group = df_lth.groupby("dataset")
    for comp, e in zip([21.5,10.85,5.55,1.05],[20,10,5,1]):
        for name, data in lth_group:
            # print(name)
            # print(data)
            fig = plt.figure(figsize=(15,10))
            x = np.arange(250)
            y = data[ np.isclose(data.compression,comp, rtol=0.06) & data.arch_size.eq("medium") & data.patience.eq(15)]["accuracy_test"].tolist()
            y = np.array(y).reshape(250)
            if name == "mnist":
                print("------------")
                print(df_set[df_set.epsilon.eq(1) & df_set.arch_size.eq("medium") & df_set.dataset.eq(name) & df_set.workers.eq(0)]["accuracy_test"])
            y2 = df_set[df_set.epsilon.eq(e) & df_set.arch_size.eq("medium") & df_set.dataset.eq(name) & df_set.workers.eq(0)]["accuracy_test"].tolist()
            y2 = np.mean(y2, axis=0)
            plt.plot(x,y, label=f"lth: {int(100-comp)+1}% sparsity") 
            plt.plot(x,y2*100,label=f"set: e{e}") 
            plt.plot
            plt.grid(color="gray")
            lth_unpruned = data[ np.isclose(data.compression,100, rtol=0.06) & data.arch_size.eq("medium") & data.patience.eq(15)]["accuracy_test"].tolist()
            lth_unpruned = np.array(lth_unpruned).reshape(250)
            plt.plot(x, lth_unpruned, label="lth_unpruned")
            plt.legend()
            plt.savefig(f"plots/lth_set_comp/lth_vs_set_{name}_e{e}")


def plot_compare_performance_configs_Set():
    df_set = dw.load_averaged_dataframes()
    df_set = df_set.drop(columns=["arch_size","start_imp", "epsilon"])
    df = df_set[df_set.workers.eq(0)]
    df = df.drop(columns=["workers"])
    for dataset in ["cifar10", "mnist", "fashionmnist"]:
        #df = df[df.dataset.eq(dataset)]
        #df = df.drop(columns=["dataset"])
        # g = df.groupby("config")["zeta_anneal","accuracy_test"] #  & averaged.zeta_anneal.eq(False)
        labels = []
        # for label, data in g:
        #     print(label)
        #     print(data[data.zeta_anneal.eq(True)]["accuracy_test"])
        #     print(data[data.zeta_anneal.eq(False)]["accuracy_test"])
        #     print("---------------")
        #     labels.append(str(label)+"anneal zeta")
        #     plt.plot(np.arange(250), data[data.zeta_anneal.eq(True)]["accuracy_test"])
        #     labels.append(label)
        #     plt.plot(np.arange(250), data[data.zeta_anneal.eq(False)]["accuracy_test"])
        #     #plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.zeta_anneal.eq(False)]["accuracy_test"]))))
        for config in list(range(1,13)):
            print(config)
            labels.append(f"{dataset} config {config}")
            data = df[df.dataset.eq(dataset) & df.config.eq(config) & df.zeta_anneal.eq(False)]["accuracy_test"]
            if len(data) == 0:
                print(config)
                continue
            data = np.array(data.tolist()).reshape(3,250)
            data = np.squeeze(np.mean(data, axis = 0)) #df.dataset.eq(dataset) 
            plt.plot(np.arange(250), data)
            plt.grid("gray")
            plt.yscale("logit")
            # labels.append(dataset)
            # plt.plot(np.arange(250), df[df.dataset.eq(dataset) & df.zeta_anneal.eq(False)]["accuracy_test"])
            plt.legend(labels)
        plt.savefig(f"plots/{dataset}_config_comparison.png")
        plt.close()
        


if __name__ == "__main__":
    #plot_compare_performance_configs_Set()
    #plot_dataset_performance_set()
    # plot_individual_performance_config()
    #plot_compare_all_configs()
    plot_compare_set_lth()

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