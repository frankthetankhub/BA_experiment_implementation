import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re
import json


omni_dict_location = "/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/omni_dict.json"
# def parse():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--root_dir",default="" ,type=str)
#     args = parser.parse_args()
#     return args

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

def search_folder(rootdir, extension):
    #file_list = []
    path_list = []
    
    for root, directories, file in os.walk(rootdir):
        for file in file:
            if(file.endswith(extension)):
                #print(root,directories)
                #file_list.append(file)
                p = root + "/" + file
                path_list.append(p)
                
    return path_list

def extract_arrays(pathlist):
    #print(pathlist)
    loss_train=[]
    loss_test=[]
    accuracy_train=[]
    accuracy_test=[]
    train_time=[]
    for path in pathlist:
        raw = np.loadtxt(path)
        #print(raw.shape)
        loss_train.append(raw[:,0])
        loss_test.append(raw[:,1])
        accuracy_train.append(raw[:,2])
        accuracy_test.append(raw[:,3])
        train_time.append(raw[:,4])
    return [loss_train, loss_test, accuracy_train, accuracy_test, train_time]

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

def extract_avg_name(path, ext):
    paths = search_folder(path, ext)
    arrays = extract_arrays(paths)
    avgs = np.average(arrays, axis=1)
    print(avgs.shape)
    keys=["Training Loss","Testing Loss","Training Accuracy","Testing Accuracy","Training Time per Epoch"]
    save_dict = dict(zip(keys,avgs.tolist()))
    return save_dict



def extract_exp_configs(path, regex):
    paths=[]
    for root, dirs, files in os.walk(path):
        if any(regex.match(dir) for dir in dirs):
            paths.append(root)
    return paths

if __name__ == "__main__":
    #if not args.root_dir:
    # path = "/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/results/s_m_p/configs5/" #cifar10_medium.txt/
    # dict_path = path[path.find("conf"):path.find(".")].split("/")
    ext = "0.txt"
    i=0
    expression = re.compile("seed*")
    paths = extract_exp_configs("/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/results/s_m_p/", expression)
    for path in paths:
        print(path)
        save_dict = extract_avg_name(path, ext)
        #print(save_dict)
        dict_path = path[path.find("conf"):path.find(".")].split("/")
        add_to_omni_dict(save_dict,dict_path)
        # if i ==2:
        #     break
        # i+=1
    # args = parse()
    # print(args.root_dir)
    # path = args.root_dir

    # print(dict_path)
    #
    #print(save_dict)
    #add_to_omni_dict(save_dict,dict_path)
        


    # paths = search_folder(path,ext) #file_list, 
    # arrays = extract_arrays(paths)
    # #averaged = average(arrays)
    # avg2 = np.average(arrays, axis=1)
    # print(avg2.shape)
    # print(avg2[3])
    # plot(avg2)