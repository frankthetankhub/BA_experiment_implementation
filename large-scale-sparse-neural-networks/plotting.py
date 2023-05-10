import numpy as np
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

def get_averages(df):
    uniques = get_uniques(df)

def get_uniques(df:pd.DataFrame,mode="set"):
    """Returns a List of unique combinations of experimental configuration parameters"""
    if mode=="set":
        uniques = [df.dataset.unique(), df.arch_size.unique(), df.start_imp.unique(), df.epsilon.unique(),df.workers.unique(), df.zeta_anneal.unique()]
        uniques = list(itertools.product(*uniques))
        print(uniques)
        
    elif mode =="lth":
        pass
    else:
        raise
    return uniques

if __name__ == "__main__":
    path = "/media/jan/9A2CA6762CA64CD7/ba_results" #cifar10_medium.txt//configs5/ large_scale/results/s_m_p
    path_lth = "/media/jan/9A2CA6762CA64CD7/ba_results/lth/results"
    df_set, df_lth_best, df_lth_raw = dw.load_dataframes()
    print(df_set.columns)
    for val in df_set.dataset.unique():
        print(val)
    get_uniques(df_set)
    groups = df_set.groupby(["dataset","arch_size","start_imp", "epsilon", "workers", "zeta_anneal"])
    #print(groups.get_group(("mnist", "small")))
    for name,data in groups:
        print(name)
        #print(data)
        #print(type(data))
        #print(data["accuracy_test"])
        #print(type(data["accuracy_test"]))
        print(np.mean(data['accuracy_test'].tolist(), axis=0))
        #break
        
    #avg = df_set[df_set.]
    #print(df_lth_best)
    #print(df_lth_raw)
    # print("-----")
    # print(df[df["dataset"]=="mnist"])
  