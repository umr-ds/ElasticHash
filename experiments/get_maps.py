import os
import numpy as np
from numpy import genfromtxt

topks = []
result_dir = "results"
maps = {"short": {}, "long": {}, "twostage": {}}
for filename in os.listdir(result_dir):
    if filename.endswith(".csv") and filename.startswith("aps."):
        topks += [int(filename.split(".")[1])]

topks = sorted(topks)
maps = np.zeros((len(topks), 4))
for i, topk in enumerate(topks):
    maps[i, 0] = topk
    topk = str(topk)
    filename = "aps." + topk + ".csv"
    data = genfromtxt(os.path.join(result_dir, filename), delimiter=',')
    print(topk + " & " + " & ".join(map(str, np.round(np.mean(data[:, 1:], axis=0), 4))) + " \\\\")
    maps[i, 1:] = np.round(np.mean(data[:, 1:], axis=0), 4)

maps = np.swapaxes(maps, 0, 1)
print("top $k$ & " + " & ".join(map(str, map(int, maps[0]))) + " \\\\")
print("\\hline")
print("short & " + " & ".join(map(str, maps[1])) + " \\\\")
print("twostage & " + " & ".join(map(str, maps[2])) + " \\\\")
print("long & " + " & ".join(map(str, maps[3])) + " \\\\")

# print (topk + " & " + " & ".join(map(str, np.round(np.mean(data[:,1:],axis=0), 5))) + " \\\\")
