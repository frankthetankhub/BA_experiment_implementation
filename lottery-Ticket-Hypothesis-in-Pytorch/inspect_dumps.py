import numpy as np
import sys
import glob
import os

def search_folder(rootdir, extension):
    file_list = []
    for root, directories, file in os.walk(rootdir):
        for file in file:
            if(file.endswith(extension)):
                file_list.append(file)
    return file_list

if __name__ == "__main__":
    if len(sys.argv) <2:
        print("please specify a folder")
    print(sys.argv)
    dir = os.getcwd() +"/"+ sys.argv[2]
    print(dir)
    paths = search_folder(dir,sys.argv[1])
    print(paths)
    for p in paths:
        pa = dir + p
        print(pa)
        print(np.load(pa,allow_pickle=True))