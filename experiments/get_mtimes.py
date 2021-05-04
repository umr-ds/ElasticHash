import os
import numpy as np
from numpy import genfromtxt

topks = set()
result_dir = "results"
indices = set()
for filename in os.listdir(result_dir):
    if filename.endswith(".csv") and filename.startswith("times."):
        s = filename.split(".")
        indices.add(s[1])
        topks.add(int(s[2]))

topks = sorted(topks)
mtimes = np.zeros((len(topks), 4))
indices = list(indices)
topks = list(topks)
index_ids = {}
for i, topk in enumerate(topks):
    mtimes[i, 0] = topk
    topk = str(topk)
    itimes = []
    for j, index in enumerate(indices):
        filename = "times." + index + "." + topk + ".csv"
        data = genfromtxt(os.path.join(result_dir, filename), delimiter=',')
        itime = np.round(np.mean(data[:, 1:], axis=0), 2)
        itimes += [itime]
        index_ids[index] = j + 1
        print(index)
    print(topk + " & " + " & ".join(map(str, itimes)) + " \\\\")
    mtimes[i, 1:] = itimes

mtimes = np.swapaxes(mtimes, 0, 1)
print("top $k$ & " + " & ".join(map(str, map(int, mtimes[0]))) + " \\\\")
print("\\hline")
print("\multirow{2}{*}{short} & $\mu & " + " & ".join(map(str, mtimes[index_ids["es-short"]])) + " \\\\")
print("\multirow{2}{*}{two-stage} & $\mu$ & " + " & ".join(map(str, mtimes[index_ids["es-twostage"]])) + " \\\\")
print("\multirow{2}{*}{long} & $\mu$ & " + " & ".join(map(str, mtimes[index_ids["es-long"]])) + " \\\\")

# print (topk + " & " + " & ".join(map(str, np.round(np.mean(data[:,1:],axis=0), 5))) + " \\\\")
