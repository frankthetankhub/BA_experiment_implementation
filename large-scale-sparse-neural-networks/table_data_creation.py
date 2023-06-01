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
lth_compression_values = [100.0, 80.1, 64.2, 51.4, 41.2, 33.0, 26.4, 21.2, 16.9, 13.6, 10.9, 8.7, 7.0, 5.6, 4.5, 3.6, 2.9, 2.3, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6]

def training_times_single_vs_multi():
    df_set, df_lth_best_acc, df_lth_all = dw.load_dataframes()
    tt_tab = df_set.pivot_table(index="workers",columns="epsilon",values="train_time",
                                 aggfunc=lambda x: np.mean(np.array(x.tolist()))
                                )
    out = tt_tab.to_latex("tables/set/train_time_workers")


def training_times_importance_pruning():
    df_set, df_lth_best_acc, df_lth_all = dw.load_dataframes()
    tt_tab = df_set.pivot_table(index=["epsilon","dataset"],columns="start_imp",values="train_time",
                                 aggfunc=lambda x: np.mean(np.array(x.tolist()))
                                )
    out = tt_tab.to_latex("tables/set/train_time_imppruning_2")

def training_times_epsilon():
    df_set, df_lth_best_acc, df_lth_all = dw.load_dataframes()
    tt_tab = df_set.pivot_table(index="epsilon",values="train_time",
                                 aggfunc=lambda x: np.mean(np.array(x.tolist()))
                                )
    out = tt_tab.to_latex("tables/set/train_time_epsilon")
    tt_tab = df_set.pivot_table(index="epsilon",values=["train_time","accuracy_test"],
                                 aggfunc=lambda x: np.mean(np.array(x.tolist()))
                                )
    print(tt_tab)
    out = tt_tab.to_latex("tables/set/train_time_and_accuracy_epsilon")

def imp_prune_avg_time():
    df_set, df_lth_best_acc, df_lth_all = dw.load_dataframes()
    tt_tab = df_set.pivot_table(columns="start_imp",values="train_time",
                                 aggfunc=lambda x: np.mean(np.array(x.tolist()))
    )
    tt_tab.to_latex("tables/set/train_time_imp_pruning_avg")
    

def create_tabular_overview():
    df_set, df_lth_best_acc, df_lth_all = dw.load_dataframes()
    tab_set = df_set.pivot_table(index =["dataset","start_imp", "zeta_anneal","workers","arch_size"],columns=["epsilon"], values=["accuracy_test"],
                                aggfunc={"accuracy_test": lambda x: [(np.mean( np.max(np.array(x.tolist()), axis=1)) *100).round(1) ,
                                                            (np.max(np.array(x.tolist())) *100).round(1),
                                                            (np.min( np.max( (np.array(x.tolist()) ) , axis=1) ) *100).round(1)] })

    print(tab_set)
    print(tab_set.shape)
    print(type(tab_set))
    print(tab_set.columns)
    tab_set.to_latex("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/set_tabular_full.tex")

    tab_set = df_set.pivot_table(index =["dataset", "zeta_anneal","workers","arch_size"],columns=["epsilon"], values=["accuracy_test"],

                                aggfunc={"accuracy_test": lambda x: [(np.mean( np.max(np.array(x.tolist()), axis=1)) *100).round(1) ,
                                                            ((np.max(np.array(x.tolist())) *100).round(1)-(np.mean( np.max(np.array(x.tolist()), axis=1)) *100)).round(1),
                                                            ((np.min( np.max( (np.array(x.tolist()) ) , axis=1) ) *100) - (np.mean( np.max(np.array(x.tolist()), axis=1)) *100)).round(1)] })
    print(tab_set)
    print(tab_set.shape)
    print(type(tab_set))
    print(tab_set.columns)
    tab_set.to_latex("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/set_tabular.tex")

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
                                            "train_time":[lambda x:np.mean(np.sum(x.tolist(), axis=1))]})
    print(tab_set)
    print(tab_set.shape)
    print(type(tab_set))
    tab_set.to_latex("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/set_traintimes_tabular.tex")
    
    b = np.isclose(df_lth_all.compression, [21.05,10.85,5.55,1.05], atol=0.05)
    
    b = [np.isclose(df_lth_all.compression, comp) for comp in [100, 21.2,10.9,5.6,1.2]]
    b = np.any(b, axis=0)
    print(b.shape)
    df_lth_selected = df_lth_all[b]
    tab_lth_all = df_lth_selected.pivot_table(index =["dataset","patience","arch_size"],columns=["compression"], values=["accuracy"],
                                  aggfunc={
                                            "accuracy": lambda x: [(np.mean( np.max(np.array(x.tolist()), axis=1) ) ).round(1) ,
                                                            (np.max(np.array(x.tolist()))).round(1),
                                                            (np.min( np.max( (np.array(x.tolist()) ) , axis=1) )).round(1) ],
                                                })
    print(tab_lth_all)
    print(tab_lth_all.shape)
    print(type(tab_lth_all))
    tab_lth_all.to_latex("/media/jan/9A2CA6762CA64CD7/ba_results/lth/lth_performance.tex")  


if __name__ == "__main__":
    training_times_single_vs_multi()
    training_times_importance_pruning()
    training_times_epsilon()
    imp_prune_avg_time()
