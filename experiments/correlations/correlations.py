import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt

# File with raw codes as learned by model
input_list = "predictions.csv"  # CSV file containing binary string in some column

num_codes = 10000000  # Number of codes to consider for computation of correlations
code_len = 256  # Lengths of code
code_pos = 0  # Column containing code as binary string in CSV

filename = str(code_len) + "_" + str(num_codes)

# Read CSV
codes = np.empty((num_codes, code_len), dtype=np.int8)
with open(input_list) as f:
    i = 0
    for row in csv.reader(f, delimiter=','):
        if i >= num_codes:
            break
        codes[i] = [np.int8(j) for j in row[code_pos]]
        i += 1

# Count 1's and 0's
plt.figure()
plt.box(False)
plt.rcParams["figure.figsize"] = (30, 15)
counts1 = np.count_nonzero(codes > 0, axis=0)
counts0 = np.count_nonzero(codes <= 0, axis=0)
ind = np.arange(code_len)  # the x locations for the groups
p1 = plt.bar(ind, counts1, width=1.0)
p2 = plt.bar(ind, counts0, bottom=counts1, width=1.0)
plt.legend((p1[0], p2[0]), ('1\'s', '0\'s'))
plt.plot()
plt.savefig(filename + "_counts.png", bbox_inches='tight',
            pad_inches=0)

# Transpose
print(codes.size, codes.shape)
codes = codes.T

# Correlations
corr = np.corrcoef(codes)
corr = abs(np.corrcoef(codes))

plt.figure()
plt.box(False)
plt.rcParams["figure.figsize"] = (40, 40)
sns.heatmap(corr)
plt.plot()
plt.savefig(filename + "_corr.png", bbox_inches='tight',
            pad_inches=0)

corr = abs(np.corrcoef(codes))
print(corr.size, corr.shape)

with open(filename + "_corr.txt", "w") as f:
    for x, row in enumerate(np.tril(corr, -1)):
        for y, c in enumerate(row):
            if c > 0:
                f.write("\t".join(map(str, [x, y, 1 - c])) + "\n")
