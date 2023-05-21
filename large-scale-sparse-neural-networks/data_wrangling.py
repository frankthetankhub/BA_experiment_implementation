import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import json
import pandas as pd

def load_raw_dicts(base_dir="/media/jan/9A2CA6762CA64CD7/ba_results/"):
    with open((base_dir+"large_scale_raw.json"), 'r') as f:
        large_scale_raw = json.load(f)
    with open((base_dir+"lth_raw.json"), 'r') as f:
        lth_raw = json.load(f)    
    return large_scale_raw, lth_raw

def large_scale(_file, as_list=True):
    raw = np.loadtxt(_file)
    workers = re.compile(".*workers_3.*")
    #print(raw.shape)
    loss_train=raw[:,0]
    loss_test=raw[:,1]
    accuracy_train=raw[:,2]
    accuracy_test=raw[:,3]
    if workers.match(_file):
        loss_train=loss_train[:-1]
        loss_test=loss_test[:-1]
        accuracy_train=accuracy_train[:-1]
        accuracy_test=accuracy_test[:-1]
    if raw.shape[1] > 4:
        train_time=raw[:,4]
    else:
        train_time=np.zeros((250))
        #hacky solution: read out total execution time from logfile and save it in the last position of train_time
        head, _ = _file.split("process_0.txt")
        logfile = head + "logs_execution.txt"
        with open(logfile) as f:
            data = f.read().replace("\n","")
            total_time = data.split("Total execution time is ")[1]
            total_time = total_time.split(" ")[0]
        train_time[-1] = total_time
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
        df = result_paths_to_df_set(all_results,save)
    elif mode == "lth_best":
        df = result_paths_to_df_lth(all_results,save)
    elif mode == "lth_all":
        df = result_paths_to_df_lth_all(all_results,save)
    return df
    #print(all_results)

def filter_files(file,mode="set"):
    if mode=="set":
        criterias = ["0.txt"] #, "bestaccuracy.dat", "times.txt"
        #criterias = ["anneal_zeta_process_0.txt","workers_3_process_0.txt",""]
    elif mode == "lth_best":
        criterias= ["bestaccuracy.dat"]
    elif mode == "lth_all":
        criterias= [".dat"]
    else:
        print(f"Mode: {mode}, not found please specify set or lth")
        raise
    for criteria in criterias:
        if file.endswith(criteria):
            return True
    return False

def result_paths_to_df_lth_all(paths, save=False):
    #results = {}
    mnist = re.compile(".*mn.*")
    cifar = re.compile(".*cifar.*")
    lt_all = re.compile(".*lt_all_.*")
    df = pd.DataFrame(columns=["dataset","arch_size","seed","compression", "accuracy","trainloss","testloss","patience"])
    #This is a bit hacky, but is done so the rounding error of file names does not propagate into the dataframe.
    compression_values = [100.0, 80.1, 64.2, 51.4, 41.2, 33.0, 26.4, 21.2, 16.9, 13.6, 10.9, 8.7, 7.0, 5.6, 4.5, 3.6, 2.9, 2.3, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6] # [round(100*0.801**x,1) for x in range(24)] 


    print(len(paths))
    for i, path in enumerate(paths):
        dataset=None
        arch_size=None
        compression=0.0
        patience=15

        seed=999
        path_list: str=path.split("/")
        for p in path_list:
            if cifar.match(p) or mnist.match(p):
                v = p.split("_")
                dataset = v[1]
                if dataset=="mn":
                    dataset = "mnist"
                elif dataset=="fmn":
                    dataset = "fashionmnist"
                arch_size= v[2]
                if len(v) > 3:
                    patience = 50
            if p.isnumeric():
                seed = int(p)
            if lt_all.match(p):
                p = p.strip(".dat")
                v = p.split("_")
                comp_val = float(v[-1])
                #compression = min(compression_values, key=lambda x:abs(x-comp_val)) #float(v[-1])
                #print(compression)
                for comp in compression_values[::-1]:
                    if np.isclose(comp_val,comp, rtol=0.1, atol=0.0999):
                        compression = comp
                        print(f"value from file: {comp_val}")
                        print(f"assigned value: {comp}")
                        break
                name = v[-2]
                if name == "loss":
                    print(path)
                    exit()
                values = np.load(path, allow_pickle = True)
                try:
                    last_val = values[values>0][-1]
                    #print(last_val)
                    values[values == 0] = last_val
                    idx = df.index[df.seed.eq(seed) & df.dataset.eq(dataset) & df.arch_size.eq(arch_size) & np.isclose(df.compression.astype(float), compression) & df.patience.eq(patience)] #
                    #print(idx)
                    values = values.tolist()
                    #print(type(values))
                    df.at[idx[0],name] = values
                except Exception as e:
                    #print(e)
                    result={"dataset":dataset,
                        "arch_size":arch_size,
                        "seed":seed,   
                        "compression":round(compression, 1),
                        "patience":patience,
                        "accuracy": None,
                        "trainloss":None,
                        "testloss":None,
                        }
                    #print(result)
                    result[name]=values
                    s = pd.Series(result)
                    df=df.append(s,ignore_index=True)
    if save:
        results = df.to_dict(orient="index")
        with open("/media/jan/9A2CA6762CA64CD7/ba_results/lth/dataframe_dict_lth_all.json", 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return df

def result_paths_to_df_lth(paths, save=False):
    results = {}
    mnist = re.compile(".*mn.*")
    cifar = re.compile(".*cifar.*")

    
    for i, path in enumerate(paths):
        dataset=None
        arch_size=None
        patience=15
        seed=999
        path_list: str=path.split("/")
        for p in path_list:
            if cifar.match(p) or mnist.match(p):
                v = p.split("_")
                dataset = v[1]
                if dataset=="mn":
                    dataset = "mnist"
                elif dataset=="fmn":
                    dataset = "fashionmnist"
                arch_size= v[2]
                if len(v) > 3:
                    patience = 50
            if p.isnumeric():
                seed = int(p)
        best_accs = np.load(path, allow_pickle = True).tolist()
        time_data_path = path.replace("lt_bestaccuracy.dat", "lt_whole_train_prune_time.dat") 
        ite_time = np.load(time_data_path, allow_pickle = True).tolist()

        result={"dataset":dataset,
                "arch_size":arch_size,
                "seed":seed,   
                "best_accuracies":best_accs,
                "patience":patience,
                "time_per_iter": ite_time
                }
        results[i]=result
    if save:
        with open("/media/jan/9A2CA6762CA64CD7/ba_results/lth/dataframe_dict_lth_bestaccs.json", 'w') as f:
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
        start_imp=140
        epsilon=None
        zeta_anneal=False
        workers=0
        seed=999
        path_list: str=path.split("/")
        for p in path_list:
            if p.startswith("configs"):
                config=int(p.strip("configs"))
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
        else:
            epsilon=1
        if config < 5:
            start_imp=200
        elif config >8:
            start_imp=250#0
        result={"dataset":dataset,
                "arch_size":arch_size,
                "start_imp":start_imp,
                "epsilon":epsilon,
                "workers":workers,
                "zeta_anneal":zeta_anneal,
                "seed":seed,
                "config":config                
                }
        if path.endswith("_0.txt"):
            loss_train, loss_test, accuracy_train, accuracy_test, train_time = large_scale(path)
            result["accuracy_test"]=accuracy_test
            result["accuracy_train"]=accuracy_train
            result["loss_train"]=loss_train
            result["loss_test"]=loss_test
            result["train_time"]=train_time
        results[i]=result
    if save:
        with open("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/dataframe_dict.json", 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    df = pd.DataFrame.from_dict(results, orient="index")
    return df

def make_avg_df(df, mode="set", as_list=True):
    if mode =="set":
        group = ["dataset","arch_size","start_imp", "epsilon", "workers", "zeta_anneal", "config"]
        new_df = pd.DataFrame(columns=df.columns)
    elif mode == "lth_all":
        group = ["dataset","arch_size", "patience", "compression"]
        new_df = pd.DataFrame(columns=df.columns)
        new_df = new_df.drop(columns=["accuracy","trainloss","testloss"])
    elif mode =="lth_best":
        group = ["dataset","arch_size", "patience"]
        new_df = pd.DataFrame(columns=df.columns)
    else:
        raise
    groups = df.groupby(group)
    
    i = 0
    for name,data in groups:
        d = dict(zip(group,name))
        # print(name)
        # print(data)
        # print(data.columns)
        # print(data.values)
        #print(name)
        #print(data['accuracy_test'])
        #print(data['accuracy_test'].tolist())
        if mode == "set":
            # print(name)
            # print(np.array(data['accuracy_test'].tolist()))
            acc_t = np.mean(np.array(data['accuracy_test'].tolist()), axis=0)
            acc_tr = np.mean(np.array(data['accuracy_train'].tolist()), axis=0)
            loss_t = np.mean(np.array(data['loss_test'].tolist()), axis=0)
            loss_tr = np.mean(np.array(data['loss_train'].tolist()), axis=0)
            time = np.mean(np.array(data['train_time'].tolist()), axis=0)
            # acc_t = np.mean(data['accuracy_test'].tolist(), axis=0)
            # acc_tr = np.mean(data['accuracy_train'].tolist(), axis=0)
            # loss_t = np.mean(data['loss_test'].tolist(), axis=0)
            # loss_tr = np.mean(data['loss_train'].tolist(), axis=0)
            # time = np.mean(data['train_time'].tolist(), axis=0)
            if as_list:
                acc_t = acc_t.tolist()
                acc_tr = acc_tr.tolist()
                loss_t = loss_t.tolist()
                loss_tr = loss_tr.tolist()
                time = time.tolist()
            
            d["accuracy_train"] = acc_tr
            d["accuracy_test"] = acc_t
            d["train_time"] = time
            d["loss_train"] = loss_tr
            d["loss_test"] = loss_t
        elif mode=="lth_best":
            acc_t = np.mean(data['best_accuracies'].tolist(), axis=0)
            time = np.mean(data['timer_per_iter'].tolist(), axis=0)
            if as_list:
                acc_t = acc_t.tolist()
                time = time.tolist()
            d["accuracy_test"] = acc_t
            d["train_time"] = time
        elif mode =="lth_all":
            acc_t = np.mean(data['accuracy'].tolist(), axis=0) 
            loss_tr = np.mean(data["trainloss"].tolist(), axis=0)
            loss_t = np.mean(data['testloss'].tolist(), axis=0)
            if as_list:
                acc_t = acc_t.tolist()
                loss_tr = loss_tr.tolist()
                loss_t = loss_t.tolist()
            d["accuracy_test"] = acc_t
            d["loss_train"] = loss_tr
            d["loss_test"] = loss_t
            d["compression"] = round(data["compression"].iloc[0],1)  ###

        new_df = new_df.append(d,ignore_index=True)
        i+=1
    new_df = new_df.drop(columns=["seed"])
    return new_df

def load_dataframes(base_dir="/media/jan/9A2CA6762CA64CD7/ba_results"):
    df_lth_all = pd.read_json(base_dir+"/lth/dataframe_dict_lth_all.json", orient="index")
    df_lth_best_acc = pd.read_json(base_dir+"/lth/dataframe_dict_lth_bestaccs.json", orient="index")
    df_set = pd.read_json(base_dir+"/large_scale/dataframe_dict.json", orient="index")
    return df_set, df_lth_best_acc, df_lth_all


def load_averaged_dataframes(base_dir="/media/jan/9A2CA6762CA64CD7/ba_results"):
    """Loads and returns the averaged Dataframes for set and lth; set is first, lth second."""
    df_lth_all = pd.read_json(base_dir+"/lth/dataframe_dict_lth_all_averaged.json", orient="index")
    #df_lth_best_acc = pd.read_json(base_dir+"/lth/dataframe_dict_lth_bestaccs.json", orient="index")
    df_set_avg = pd.read_json(base_dir+"/large_scale/dataframe_dict_averaged.json", orient="index")
    return df_set_avg, df_lth_all#, df_lth_best_acc

def create_all_dataframes(base_dir="/media/jan/9A2CA6762CA64CD7/ba_results", save = False, save_non_averaged=False, specs = ["set","lth_all","lth_best"]):
    for spec in specs: #
        print(f"currently doing spec: {spec}")
        df = get_data_as_dataframe(base_dir, mode=spec, save=save_non_averaged)
        df_avg = make_avg_df(df, mode=spec)
        print("df averaged:")
        #print(df_avg)
        print(df_avg.iloc[0])
        print(df_avg.iloc[1])
        dic = df_avg.to_dict(orient="index")
        print(dic[0])
        if spec == "set":
            save_path="/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/dataframe_dict_averaged.json"
        elif spec == "lth_all":
            save_path = "/media/jan/9A2CA6762CA64CD7/ba_results/lth/dataframe_dict_lth_all_averaged.json"
        else:
            save_path = "/media/jan/9A2CA6762CA64CD7/ba_results/lth/dataframe_dict_lth_bestaccs_averaged.json"
        if save:
            with open(save_path, 'w') as f:
                json.dump(dic, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # df, _ ,__ = load_dataframes()
    #make_avg_df(df)
    create_all_dataframes(save=True, save_non_averaged = True) #, specs=["lth_all"]

    # path = "/media/jan/9A2CA6762CA64CD7/ba_results" #cifar10_medium.txt//configs5/ large_scale/results/s_m_p
    # df_raw_set = get_data_as_dataframe(path,True)
    # #print(df_raw_set)
    # df_avg_set = make_avg_df(df_raw_set)
    # #print(df_avg_set)
    # json_file = df_avg_set.to_json()
    # dic = df_avg_set.to_dict(orient="index")
    #print(dic)
    #print(json_file)
    # with open("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/dataframe_dict_averaged.json", 'w') as f:
    #         #json.dump(json_file, f, ensure_ascii=False, indent=2)
    #         #f.write(json_file)
    #         json.dump(dic, f, ensure_ascii=False, indent=2)

    # df_set_avg = load_averaged_dataframes()#
    # print(df_set_avg)
    # df_set_avg.head()