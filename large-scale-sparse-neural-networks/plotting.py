import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir",default="" ,type=str)
    args = parser.parse_args()
    return args

def search_folder(rootdir, extension):
    file_list = []
    path_list = []
    
    for root, directories, file in os.walk(rootdir):
        for file in file:
            if(file.endswith(extension)):
                print(root,directories)
                file_list.append(file)
                p = root + "/" + file
                path_list.append(p)
                
    return file_list, path_list

def average(metric_list):
    #print(metric_list)
    #arr = np.array(metric_list)
    #print(arr.shape)
    averages=[]
    for metric in metric_list:
        # for array in arrays:
        #     print(array.shape)
        #     print(array[0])
        #     pass
        arr = np.array(metric)
        print(arr)
        print(arr.shape)
        avg = np.average(arr, axis = 0)
        print(avg.shape)
        averages.append(avg)
    print(averages)
    print(len(averages))
         

def extract_arrays(pathlist):
    print(pathlist)
    loss_train=[]
    loss_test=[]
    accuracy_train=[]
    accuracy_test=[]
    for path in pathlist:
        raw = np.loadtxt(path)
        print(raw.shape)
        print(type(raw))
        loss_train.append(raw[:,0])
        loss_test.append(raw[:,1])
        accuracy_train.append(raw[:,2])
        accuracy_test.append(raw[:,3])
        # print(raw.dtype)
        # print(raw.shape)
        # print(accuracy_test)
        # print(accuracy_train)
        # break
    return [loss_train, loss_test, accuracy_train, accuracy_test]

def plot(avg2):
    i=0
    metrics = ["Training Loss","Testing Loss","Training Accuracy","Testing Accuracy"]
    for metric in avg2:
        
        l=metric.shape[0]
        print(l)
        a = np.arange(l)
        plt.plot(a, metric, c="blue", label="Winning tickets") 
        # plt.plot(a, c, c="red", label="Random reinit") 
        t = metrics[i]
        print(t)
        plt.title(t) #Test Accuracy vs Weights % ({arch_type} | {dataset})
        # plt.xlabel("Weights %") 
        # plt.ylabel("Test accuracy") 
        plt.xticks(np.arange(l,step=10), rotation ="vertical") 
        # plt.ylim(0,100)
        # plt.legend() 
        # plt.grid(color="gray") 

        plt.savefig(f"{os.getcwd()}/plots/{i}.png", dpi=1200, bbox_inches='tight') 
        #plt.show()
        plt.close()
        i+=1

if __name__ == "__main__":
    args = parse()
    print(args.root_dir)
    path = args.root_dir
    if not args.root_dir:
        path = "/media/jan/9A2CA6762CA64CD7/ba_results/large_scale/results/s_m_p/configs5/cifar10_small.txt/"
        ext = "0.txt"
        
    file_list, paths = search_folder(path,ext)
    arrays = extract_arrays(paths)
    #averaged = average(arrays)
    avg2 = np.average(arrays, axis=1)
    print(avg2.shape)
    print(avg2[3])
    plot(avg2)
    