import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute correlations between bit positions')

    parser.add_argument(
        '--input_list',
        default="images.with.codes.csv",
        type=str,
        help='CSV file containing binary string in some column'
    )

    parser.add_argument(
        '--output_dir',
        default=".",
        type=str,
        help='Write correlation files to this directory'
    )

    parser.add_argument(
        '--num_codes',
        default=10000000,
        type=int,
        help='Maximum number of codes to consider for computation of correlations'
    )

    parser.add_argument(
        '--code_len',
        default=256,
        type=int,
        help='Lengths of code'
    )

    parser.add_argument(
        '--code_pos',
        default=1,
        type=int,
        help='Column containing code as binary string in CSV'
    )

    args = parser.parse_args()

    input_list = args.input_list  # CSV file containing binary string in some column
    output_dir = args.output_dir
    num_codes = args.num_codes  # Number of codes to consider for computation of correlations
    code_len = args.code_len  # Lengths of code
    code_pos = args.code_pos  # Column containing code as binary string in CSV

    filename = str(code_len)  # + "_" + str(num_codes)

    # Read CSV
    with open(input_list) as f:
        linecount = len(f.readlines())
        num_codes = min(linecount, num_codes)
        codes = np.empty((num_codes, code_len), dtype=np.int8)
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
    plt.savefig(os.path.join(output_dir, filename + "_counts.png"), bbox_inches='tight',
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
    plt.savefig(os.path.join(output_dir, filename + "_corr.png"), bbox_inches='tight',
                pad_inches=0)

    corr = abs(np.corrcoef(codes))
    print(corr.size, corr.shape)

    with open(os.path.join(output_dir, filename + "_corr.txt"), "w") as f:
        for x, row in enumerate(np.tril(corr, -1)):
            for y, c in enumerate(row):
                if c > 0:
                    f.write("\t".join(map(str, [x, y, 1 - c])) + "\n")
