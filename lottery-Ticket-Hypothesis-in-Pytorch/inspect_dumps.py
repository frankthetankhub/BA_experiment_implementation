import numpy as np
import sys
import glob
import os

if __name__ == "__main__":
    if len(sys.argv) <2:
        print("please specify a folder")
    print(sys.argv)
    dir = os.getcwd() +"/"+ sys.argv[2]
    print(dir)
    paths = glob.glob(sys.argv[1], root_dir=dir)
    print(paths)
    for p in paths:
        print(p)
        print(np.load(path,allow_pickle=True))