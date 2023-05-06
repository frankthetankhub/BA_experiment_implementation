import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import json
import pandas as pd

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

def large_scale(file, as_list=True):
    raw = np.loadtxt(file)
    #print(raw.shape)
    loss_train=raw[:,0]
    loss_test=raw[:,1]
    accuracy_train=raw[:,2]
    accuracy_test=raw[:,3]
    if raw.shape[1] > 4:
        train_time=raw[:,4]
    else:
        train_time=np.zeros((250))
    if as_list:
        loss_train=loss_train.tolist()
        loss_test=loss_test.tolist()
        accuracy_train=accuracy_train.tolist()
        accuracy_test=accuracy_test.tolist()
        train_time=train_time.tolist()
    return loss_train, loss_test, accuracy_train, accuracy_test, train_time

def get_data_as_dataframe(exp_root_path, save=False, mode="set"):
    #large_scale_raw, lth_raw = load_raw_dicts()
    all_results = []
    for root, dirs, files in os.walk(exp_root_path):
        for file in files:
            if filter_files(file, mode):
                all_results.append(root+"/"+file)
    if mode =="set":
        result_paths_to_df_set(all_results,save)
    elif mode == "lth":
        result_paths_to_df_lth(all_results,save)
    #print(all_results)

def filter_files(file,mode="set"):
    if mode=="set":
        criterias = ["0.txt"] #, "bestaccuracy.dat", "times.txt"
    elif mode == "lth":
        criteria= [".dat"]
    else:
        print(f"Mode: {mode}, not found please specify set or lth")
        raise
    for criteria in criterias:
        if file.endswith(criteria):
            return True
    return False

def result_paths_to_df_lth(paths, save=False):
    results = {}
    mnist = re.compile(".*mnist.*")
    cifar = re.compile(".*cifar:*")
    zeta = re.compile(".*anneal_*")
    multi = re.compile(".*workers_3.*")
    
    for i, path in enumerate(paths):
        dataset=None
        arch_size=None
        compression=100

        seed=999
        path_list: str=path.split("/")
        for p in path_list:
            if p.startswith("config"):
                config=int(p[-1])
            if cifar.match(p) or mnist.match(p):
                v = p.split("_")
                dataset = v[0]
                arch_size= v[1].rstrip(".txt")
            if p.startswith("seed"):
                seed = int(p.lstrip("seed_"))
            try:
                seed = int(p)
            except ValueError:
                pass

        result={"dataset":dataset,
                "arch_size":arch_size,
                "seed":seed,                
                }
        if path.endswith("_0.txt"):
            loss_train, loss_test, accuracy_train, accuracy_test, train_time = large_scale(path)
            result["loss_train"]=loss_train
            result["loss_test"]=loss_test
            result["accuracy_train"]=accuracy_train
            result["accuracy_test"]=accuracy_test
            result["train_time"]=train_time
        results[i]=result
        # if i == 100:
        #     break
    #print(results)
    if save:
        with open("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/dataframe_dict.json", 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    df = pd.DataFrame.from_dict(results, orient="index")
    return df

def result_paths_to_df_set(paths, save=False):
    results = {}
    mnist = re.compile(".*mnist.*")
    cifar = re.compile(".*cifar:*")
    zeta = re.compile(".*anneal_*")
    multi = re.compile(".*workers_3.*")
    
    for i, path in enumerate(paths):
        dataset=None
        arch_size=None
        start_imp=0
        epsilon=None
        zeta_anneal=False
        workers=0
        seed=999
        path_list: str=path.split("/")
        for p in path_list:
            if p.startswith("config"):
                config=int(p[-1])
            if cifar.match(p) or mnist.match(p):
                v = p.split("_")
                dataset = v[0]
                arch_size= v[1].rstrip(".txt")
            if p.startswith("seed"):
                seed = int(p.lstrip("seed_"))
            if zeta.match(p):
                zeta_anneal=True
            if multi.match(p):
                workers=3
        if config%4==1:
            epsilon=20
        elif config%4==2:
            epsilon=10
        elif config%4==3:
            epsilon=5
        else: #if config%4==0:
            epsilon=1
        if config < 5:
            start_imp=200
        elif config >8:
            start_imp=140
        result={"dataset":dataset,
                "arch_size":arch_size,
                "start_imp":start_imp,
                "epsilon":epsilon,
                "workers":workers,
                "zeta_anneal":zeta_anneal,
                "seed":seed,                
                }
        if path.endswith("_0.txt"):
            loss_train, loss_test, accuracy_train, accuracy_test, train_time = large_scale(path)
            result["loss_train"]=loss_train
            result["loss_test"]=loss_test
            result["accuracy_train"]=accuracy_train
            result["accuracy_test"]=accuracy_test
            result["train_time"]=train_time
        results[i]=result
        # if i == 100:
        #     break
    #print(results)
    if save:
        with open("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/dataframe_dict.json", 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    df = pd.DataFrame.from_dict(results, orient="index")
    return df


        


if __name__ == "__main__":
    path = "/media/jan/9A2CA6762CA64CD7/ba_results" #cifar10_medium.txt//configs5/ large_scale/results/s_m_p
    #combined(path, ".txt")
    # TODO:
    # create df with raw data instead of dictornary
    get_data_as_dataframe(path, save=True)
    data = pd.read_json("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/dataframe_dict.json",orient="index")
    print(data.isnull().sum())
    #print(data[data["dataset"]=="mnist"]["accuracy_train"])




def add_to_omni_dict(save_dict, dict_path):
    with open(omni_dict_location, 'r') as f:
        omni_dict = json.load(f)
    #print(omni_dict.keys())
    try:
        d = omni_dict[dict_path[0]]
    except KeyError:
        print(f"KEY: {dict_path[0]} does not exist and gets added now")
        omni_dict[dict_path[0]]={}
        d = omni_dict[dict_path[0]]
    dict_path = dict_path[1:]
    print(d.keys())
    for depth in dict_path[:-1]:
        try:
            d = d[depth]
        except KeyError:
            print(f"KEY: {depth} does not exist and gets added now")
            d[depth]={}
            d = d[depth]
        #print(d.keys())
    d[dict_path[-1]]=save_dict
    with open(omni_dict_location, 'w') as f:
        json.dump(omni_dict, f, ensure_ascii=False, indent=2)