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
    plt.figure(figsize=(15,10))
    for label, data in g:
        print(data)
        labels.append(label+"anneal zeta")
        plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.zeta_anneal.eq(True)]["accuracy_test"].tolist()),axis=0)))
        labels.append(label)
        plt.plot(np.arange(250), np.squeeze(np.mean(np.array(data[data.zeta_anneal.eq(False)]["accuracy_test"].tolist()),axis=0)))
    plt.legend(labels)
    plt.savefig("plots/dataset_performance_averaged_anneal_comparison.png")

def plot_dataset_performance_set():
    averaged, _ = dw.load_averaged_dataframes()
    #averaged = dw.make_avg_df(df_set)
    g = averaged[averaged.workers.eq(0)].groupby(["dataset","arch_size"])["zeta_anneal","accuracy_test"] #  & averaged.zeta_anneal.eq(False)
    labels = []
    plt.figure(figsize=(15,10))
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

def plot_non_averaged():
    df_set, df_lth_best_acc, df_lth_all = dw.load_dataframes()
    vals = df_set[df_set.epsilon.eq(1) & df_set.arch_size.eq("medium") & df_set.dataset.eq("cifar10") & df_set.workers.eq(0) & df_set.zeta_anneal.eq(False) & df_set.start_imp.eq(0)]#["accuracy_test"]
    print(vals.columns)
    print(vals)
    print(vals["seed"])
    for v,seed in zip(vals.accuracy_test, vals.seed):
        
        plt.plot(np.arange(250), 100*np.array(v), label=f"SET, seed:{seed}")
    es = df_lth_all[np.isclose(df_lth_all.compression,1.05, atol=0.05) & df_lth_all.arch_size.eq("medium") & df_lth_all.dataset.eq("cifar10") & df_lth_all.patience.eq(15)]
    print(es)
    print(es.columns)
    print(es.seed)
    i=0
    for v,seed in zip(es.accuracy,es.seed):
        # if i>1: 
        #     break
        plt.plot(np.arange(250), v, label=f"LTH, seed:{seed}", linewidth=(10-(i*2)))
        i+=1
    plt.legend()
    plt.savefig("plots/teststet")
    plt.close()

def plot_compare_single_multi_worker():
    df_set, df_lth_best_acc, df_lth_all = dw.load_dataframes()
    groups = df_set[df_set.zeta_anneal.eq(False)].groupby(["dataset", "arch_size", "config"])
    for name, data in groups:
        multi = data[data.workers.eq(3)]
        single = data[data.workers.eq(0)]
        for index,row in multi.iterrows():
            #print(index)
            #print(row["accuracy"])
            plt.plot(row["accuracy_test"], alpha=0.2, color="blue", linewidth=1, label=f"WASAP, seed:{row.seed}")
        for index,row in single.iterrows():
            #print(index)
            #print(row["accuracy"])
            plt.plot(row["accuracy_test"], color="orange", alpha=0.2, linewidth=1 ,label=f"SET, seed:{row.seed}")
        plt.plot(np.average(np.array(single["accuracy_test"].tolist()), axis=0), color="orange", linewidth=1 ,label=f"Single Worker, average")
        plt.plot(np.average(np.array(multi["accuracy_test"].tolist()), axis=0), color="blue", linewidth=1 ,label=f"WASAP, average")
        plt.legend()
        plt.xlabel("Training Epochs")
        plt.ylabel("Test Accuracy")
        plt.suptitle(f"Performance Comparison of Single Worker and WASAP:\n\nConfig: {name[2]}          Dataset: {name[0].capitalize()}          Architecture Size: {name[1].capitalize()}")
        #plt.suptitle(f"Performance Comparison of Single Worker and WASAP:\nConfig: {name[2]}\nDataset: {name[0].capitalize()}\n Architecture Size: {name[1].capitalize()}")
        plt.gcf().set_size_inches(15, 8)
        plt.xlim([-0.5, 250])
        plt.tight_layout()
        plt.savefig(f"plots/num_worker_comparison/{name[0]}_{name[1]}_{name[2]}")
        plt.close("all")
    for name, data in df_set[df_set.zeta_anneal.eq(False)].groupby(["dataset", "arch_size"]):
        print(name)
        multi = data[data.workers.eq(3)]
        single = data[data.workers.eq(0)] 
        avg_multi=np.average(np.array(multi["accuracy_test"].tolist()),axis=0)
        plt.plot(avg_multi, color="blue", linewidth=1 ,label=f"WASAP")
        plt.plot(np.average(np.array(single["accuracy_test"].tolist()), axis=0), color="orange", linewidth=1 ,label=f"Single Worker")#3 Workers,averaged over all configs
        #if name[0]=="cifar10":
        plt.hlines(np.max(avg_multi), 0,250, color="blue", alpha=0.5, ls="--")
        plt.legend()
        plt.xlabel("Training Epochs")
        plt.ylabel("Test Accuracy")
        #plt.suptitle(f"Performance Comparison of Single Worker and WASAP, Averaged Over All Configurations\nDataset: {name[0].capitalize()}\n Architecture Size: {name[1].capitalize()}")
        plt.suptitle(f"Performance Comparison of Single Worker and WASAP, Averaged Over All Configurations\n\nDataset: {name[0].capitalize()}          Architecture Size: {name[1].capitalize()}")
        plt.gcf().set_size_inches(15, 8)
        plt.xlim([-0.5, 250])
        plt.tight_layout()
        plt.savefig(f"plots/num_worker_comparison/{name[0]}_{name[1]}_averaged")
        plt.close("all")

def plot_all_lth_raw():
    df_set, df_lth_best_acc, df_lth_all = dw.load_dataframes()   
    print(df_lth_all.compression) 
    lth_groups = df_lth_all.groupby(["arch_size","dataset","compression"])
    for name,data in lth_groups:
        #print(data)
        #print(type(data))
        print(name)
        b = np.isclose(name[2], [21.05,10.85,5.55,1.05], atol=0.05)#.any()
        print(b)
        if not b.any():
            # print(name)
            # print(data)
            continue
        pat_15 = data[data.patience.eq(15)]# , np.isclose(data.compression,data, rtol=0.06)
        pat_50 = data[data.patience.eq(50)]
        #print(pat_15)
        #print(pat_50)
        #fig = plt.Figure(figsize=(25,10))
        for index,row in pat_15.iterrows():
            #print(index)
            #print(row["accuracy"])
            plt.plot(row["accuracy"], alpha=0.3, color="blue", linewidth=1, label=f"Patience: 15, seed:{row.seed}")
            lth_treshold = np.max(df_lth_all[df_lth_all.patience.eq(15) & df_lth_all.seed.eq(row.seed) & df_lth_all.dataset.eq(name[1]) & df_lth_all.arch_size.eq(name[0]) & df_lth_all.compression.eq(100)]["accuracy"].tolist())
            plt.hlines(lth_treshold,0,250, colors="gray", ls="--", linewidth=1)
            comp100= df_lth_all[df_lth_all.patience.eq(15) &df_lth_all.seed.eq(row.seed) & df_lth_all.dataset.eq(name[1]) & df_lth_all.arch_size.eq(name[0]) & df_lth_all.compression.eq(100)]["accuracy"].tolist()
            #print(comp100)
            plt.plot(comp100[0], color = "gray",label =f"Patience: 15, seed:{row.seed}, unpruned")

        for index,row in pat_50.iterrows():
            #print(index)
            #print(row["accuracy"])
            plt.plot(row["accuracy"], color="orange", alpha=0.3, linewidth=1 ,label=f"Patience: 50, seed:{row.seed}")
            lth_treshold = np.max(df_lth_all[df_lth_all.patience.eq(50) &df_lth_all.seed.eq(row.seed) & df_lth_all.dataset.eq(name[1]) & df_lth_all.arch_size.eq(name[0]) & df_lth_all.compression.eq(100)]["accuracy"].tolist())
            plt.hlines(lth_treshold,0,250, colors="gray", ls="--", linewidth=1)
            comp100= df_lth_all[df_lth_all.patience.eq(50) &df_lth_all.seed.eq(row.seed) & df_lth_all.dataset.eq(name[1]) & df_lth_all.arch_size.eq(name[0]) & df_lth_all.compression.eq(100)]["accuracy"].tolist()
            #print(comp100)
            plt.plot(comp100[0], color = "gray",label =f"Patience: 50, seed:{row.seed}, unpruned")
        #plt.plot(pat_50, color="red")
        save_comp=str(name[2]).replace('.',"_")
        plt.legend()
        plt.gcf().set_size_inches(15, 8)
        plt.savefig(f"plots/individual_runs_variance/lth/{name[1]}_{name[0]}_{save_comp}")
        plt.close("all")




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
    plt.close("all")


def plot_compare_set_lth2():
    df_set, df_lth = dw.load_averaged_dataframes()
    datasets = df_lth.groupby("dataset")
    for comp, e in zip([21.4,10.85,5.55,1.05],[20,10,5,1]):
        plt.rc("font", size=18)
        for name, data in datasets:

            #fig, ax = plt.subplots(1,3)
            #for arch_size in ["small", "medium", "large"]:
            #print(arch_size)
            plt.figure(figsize=(15,10))
            plt.suptitle(f"Lottery Ticket vs. Sparse Evolutionary Training \n {name}".title())
            x = np.arange(250)
            plt.xlabel("Epochs")
            plt.ylabel("Test Accuracy")
            plt.plot(x, 100*np.mean(df_set[df_set.epsilon.eq(e) & df_set.arch_size.eq("small") & df_set.dataset.eq(name) & df_set.workers.eq(0)]["accuracy_test"].tolist(), axis=0),label=f"set: e{e} small")  
            plt.plot(x, 100*np.mean(df_set[df_set.epsilon.eq(e) & df_set.arch_size.eq("medium") & df_set.dataset.eq(name) & df_set.workers.eq(0)]["accuracy_test"].tolist(), axis=0),label=f"set: e{e} medium")
            plt.plot(x, 100*np.mean(df_set[df_set.epsilon.eq(e) & df_set.arch_size.eq( "large") & df_set.dataset.eq(name) & df_set.workers.eq(0)]["accuracy_test"].tolist(), axis=0),label=f"set: e{e} large")
            plt.plot(x,np.array(data[ np.isclose(data.compression,comp, rtol=0.06) & data.arch_size.eq("small") & data.patience.eq(15)]["accuracy_test"].tolist()).reshape(250), label=f"lth: {int(100-comp)+1}% sparsity small")
            plt.plot(x,np.array(data[ np.isclose(data.compression,comp, rtol=0.06) & data.arch_size.eq("medium") & data.patience.eq(15)]["accuracy_test"].tolist()).reshape(250), label=f"lth: {int(100-comp)+1}% sparsity medium")
            plt.plot(x,np.array(data[ np.isclose(data.compression,comp, rtol=0.06) & data.arch_size.eq( "large") & data.patience.eq(15)]["accuracy_test"].tolist()).reshape(250), label=f"lth: {int(100-comp)+1}% sparsity large") 


            
            lth_unpruned = data[ np.isclose(data.compression,100, rtol=0.06) & data.arch_size.eq("small") & data.patience.eq(15)]["accuracy_test"].tolist()
            lth_unpruned = np.array(lth_unpruned).reshape(250)
            max_acc = np.max(lth_unpruned)
            plt.plot(x, lth_unpruned, label=f"lth_unpruned small architecture", color="aqua")
            plt.hlines(y=max_acc, xmax=250, xmin=0, ls="--", colors="aqua", alpha=0.4)
            lth_unpruned = data[ np.isclose(data.compression,100, rtol=0.06) & data.arch_size.eq("medium") & data.patience.eq(15)]["accuracy_test"].tolist()
            lth_unpruned = np.array(lth_unpruned).reshape(250)
            max_acc = np.max(lth_unpruned)
            plt.plot(x, lth_unpruned, label=f"lth_unpruned medium architecture", color="steelblue")
            plt.hlines(y=max_acc, xmax=250, xmin=0, ls="--", colors="steelblue", alpha=0.4)
            lth_unpruned = data[ np.isclose(data.compression,100, rtol=0.06) & data.arch_size.eq("large") & data.patience.eq(15)]["accuracy_test"].tolist()
            lth_unpruned = np.array(lth_unpruned).reshape(250)
            max_acc = np.max(lth_unpruned)
            plt.plot(x, lth_unpruned, label=f"lth_unpruned large architecture", color="lightgray")
            plt.hlines(y=max_acc, xmax=250, xmin=0, ls="--", colors="lightgray", alpha=0.4)
            plt.legend()
            plt.grid(color="gray")
            plt.savefig(f"plots/lth_set_comp/lth_vs_set_{name}_e{e}")
            plt.close("all")

def plot_compare_performance_configs_Set():
    df_set = dw.load_averaged_dataframes()
    df_set = df_set.drop(columns=["arch_size","start_imp", "epsilon"])
    df = df_set[df_set.workers.eq(0)]
    df = df.drop(columns=["workers"])
    for dataset in ["cifar10", "mnist", "fashionmnist"]:
        labels = []

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
        
def create_tabular_overview():
    df_set, df_lth_best_acc, df_lth_all = dw.load_dataframes()
    tab_set = df_set.pivot_table(index =["dataset","start_imp", "zeta_anneal","workers","arch_size"],columns=["epsilon"], values=["accuracy_test"], #, "train_time"
                                #   aggfunc={"accuracy_test": [lambda x: (np.mean( np.max(np.array(x.tolist()), axis=1)) *100).round(1) ,
                                #                             lambda x: (np.max(np.array(x.tolist())) *100).round(1),
                                #                             lambda x: (np.min( np.max( (np.array(x.tolist()) ) , axis=1) ) *100).round(1)] })
                                aggfunc={"accuracy_test": lambda x: [(np.mean( np.max(np.array(x.tolist()), axis=1)) *100).round(1) ,
                                                            (np.max(np.array(x.tolist())) *100).round(1),
                                                            (np.min( np.max( (np.array(x.tolist()) ) , axis=1) ) *100).round(1)] }) #,
                                            #                                                           "train_time":[lambda x:np.mean(np.sum(x.tolist(), axis=0))]
    print(tab_set)
    print(tab_set.shape)
    print(type(tab_set))
    print(tab_set.columns)
    tab_set.to_latex("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/set_tabular_full.tex")

    tab_set = df_set.pivot_table(index =["dataset", "zeta_anneal","workers","arch_size"],columns=["epsilon"], values=["accuracy_test"],

                                aggfunc={"accuracy_test": lambda x: [(np.mean( np.max(np.array(x.tolist()), axis=1)) *100).round(1) ,
                                                            ((np.max(np.array(x.tolist())) *100).round(1)-(np.mean( np.max(np.array(x.tolist()), axis=1)) *100)).round(1),
                                                            ((np.min( np.max( (np.array(x.tolist()) ) , axis=1) ) *100) - (np.mean( np.max(np.array(x.tolist()), axis=1)) *100)).round(1)] })
                                # aggfunc={"accuracy_test": lambda x: [(np.mean( np.max(np.array(x.tolist()), axis=1)) *100).round(1) ,
                                #                         (np.max(np.array(x.tolist())) *100).round(1)-(np.mean( np.max(np.array(x.tolist()), axis=1)) *100).round(1),
                                #                         (np.min( np.max( (np.array(x.tolist()) ) , axis=1) ) *100).round(1) - (np.mean( np.max(np.array(x.tolist()), axis=1)) *100).round(1)] })
    print(tab_set)
    print(tab_set.shape)
    print(type(tab_set))
    print(tab_set.columns)
    tab_set.to_latex("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/set_tabular.tex")
    
    print(df_lth_best_acc)

    #test for correctness
    # for name, data in df_set.groupby(["dataset","start_imp","workers", "zeta_anneal", "arch_size","epsilon"]):
    #     print(name)
    #     print(data.train_time)
    #     print(data.iloc[0].train_time)
    #     print(np.sum(data.iloc[0].train_time))
    #     print(data.iloc[1].train_time)
    #     print(np.sum(data.iloc[1].train_time))
    #     print(data.iloc[2].train_time)
    #     print(np.sum(data.iloc[2].train_time))
    #     print(np.mean(np.sum(data.train_time.tolist(), axis=1)))
    #     return
    tab_set = df_set.pivot_table(index =["dataset","start_imp","workers", "arch_size"],columns=["epsilon"], values=[ "train_time"],
                                  aggfunc={
                                            # "accuracy_test": [lambda x: (np.mean( np.max(np.array(x.tolist()), axis=0)) *100).round(1) ,
                                            #                 lambda x: (np.max(np.array(x.tolist())) *100).round(1),
                                            #                 lambda x: (np.min( np.max( (np.array(x.tolist()) ) , axis=1) ) *100).round(1) ],
                                            "train_time":[lambda x:np.mean(np.sum(x.tolist(), axis=1))]}) #"config" ,lambda x: np.min(np.array(x.tolist())) ,"epsilon","zeta_anneal" , axis=1
    print(tab_set)
    print(tab_set.shape)
    print(type(tab_set))
    tab_set.to_latex("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/set_traintimes_tabular.tex")

    tab_set = df_set.pivot_table(index =["dataset","start_imp","workers"],columns=["epsilon"], values=[ "train_time"], #, "arch_size"
                                  aggfunc={
                                            # "accuracy_test": [lambda x: (np.mean( np.max(np.array(x.tolist()), axis=0)) *100).round(1) ,
                                            #                 lambda x: (np.max(np.array(x.tolist())) *100).round(1),
                                            #                 lambda x: (np.min( np.max( (np.array(x.tolist()) ) , axis=1) ) *100).round(1) ],
                                            "train_time":[lambda x:np.mean(np.sum(x.tolist(), axis=1))]}) #"config" ,lambda x: np.min(np.array(x.tolist())) ,"epsilon","zeta_anneal" , axis=1
    print(tab_set)
    print(tab_set.shape)
    print(type(tab_set))
    tab_set.to_latex("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/set_traintimes_tabular_avg_over_archsize.tex")

def plot_anneal_epsilon_comparison():
    df_set, _ = dw.load_averaged_dataframes()
    #df_set = df_set.drop(columns=["start_imp"])
    for name,data in df_set.groupby(["epsilon", "arch_size"]):
        for dataset in ["cifar10","fashionmnist","mnist"]:
            #print(data[data.dataset.eq(dataset) & data.workers.eq(0) & data.zeta_anneal.eq(True)& data.start_imp.eq(0)].accuracy_test.tolist()[0])
            plt.plot(data[data.dataset.eq(dataset) & data.workers.eq(0) & data.zeta_anneal.eq(True)& data.start_imp.eq(0)].accuracy_test.tolist()[0],label="anneal: True")
            plt.plot(data[data.dataset.eq(dataset) & data.workers.eq(0) & data.zeta_anneal.eq(False)& data.start_imp.eq(0)].accuracy_test.tolist()[0],label="anneal: False")
            plt.legend()
            plt.savefig(f"plots/anneal_comp/{dataset}/{name}")
            plt.close("all")

if __name__ == "__main__":
    #plot_compare_performance_configs_Set()
    #plot_dataset_performance_set()
    # plot_individual_performance_config()
    #plot_compare_all_configs()
    # plot_dataset_performance_averaged_set()
    # plot_dataset_performance_set()
    # plot_compare_set_lth2()
    #plot_non_averaged()
    #plot_all_lth_raw()
    #plot_compare_single_multi_worker()
    create_tabular_overview()
    #plot_anneal_epsilon_comparison()
