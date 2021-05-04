import argparse
import os.path
from util import uint2binstr, int2binstr, binstr2int, binstr2uint, permute_code
import csv
import pickle as pkl

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create long and short codes from 2-stage csv')

    parser.add_argument(
        '--csv',
        required=True,
        type=str,
        help='Path to input list containing image pathes and codes'
    )

    parser.add_argument(
        '--sep',
        default=',',
        type=str,
        help='Separator for image list (default: \',\')'
    )

    parser.add_argument(
        '--col_codes',
        default=3,
        type=int,
        help='First column id of codes (starts with 0, default: 3)'
    )

    args = parser.parse_args()

    code_cols_start = args.col_codes

    # Load correlations
    with open("correlations/256_16_perm.pkl", "rb") as f:
        perm_256_16, _ = pkl.load(f)

    # Read CSV
    csv_file = open(args.csv, "r")
    csv_reader = csv.reader(csv_file, delimiter=args.sep)

    outpath = os.path.splitext(args.csv)[0]
    out_short = outpath + ".short.csv"
    out_long64 = outpath + ".long64.csv"
    out_long = outpath + ".long.csv"

    csv_file_short = open(out_short, "w")
    csv_file_long64 = open(out_long64, "w")
    csv_file_long = open(out_long, "w")

    csv_writer_short = csv.writer(csv_file_short, delimiter=args.sep)
    csv_writer_long = csv.writer(csv_file_long, delimiter=args.sep)
    csv_writer_long64 = csv.writer(csv_file_long64, delimiter=args.sep)

    for row in csv_reader:
        prec = row[:code_cols_start]
        succ = row[code_cols_start + 8:]

        # signed int
        f0 = int(row[code_cols_start + 0])
        f1 = int(row[code_cols_start + 1])
        f2 = int(row[code_cols_start + 2])
        f3 = int(row[code_cols_start + 3])

        # unsigned int
        r0 = int(row[code_cols_start + 4])
        r1 = int(row[code_cols_start + 5])
        r2 = int(row[code_cols_start + 6])
        r3 = int(row[code_cols_start + 7])

        # Convert short filter subcodes (16 bit unsigned int) to binary string and then to subcodes (16 bit signed int)
        short_r0 = binstr2int(
            uint2binstr(f0, length=16) + uint2binstr(f1, length=16) + uint2binstr(f2, length=16) + uint2binstr(f3,
                                                                                                               length=16))

        # Convert long subcodes (64 bit signed int) to binary string and then to filter subcodes (64 bit unsigned int)
        long64_f0 = binstr2uint(int2binstr(r0, length=64))
        long64_f1 = binstr2uint(int2binstr(r1, length=64))
        long64_f2 = binstr2uint(int2binstr(r2, length=64))
        long64_f3 = binstr2uint(int2binstr(r3, length=64))

        #
        str_256 = int2binstr(r0, length=64) + int2binstr(r1, length=64) + int2binstr(r2, length=64) + int2binstr(r3,
                                                                                                                 length=64)
        str_256_permuted = permute_code(str_256, perm_256_16)
        long_f = []
        num_subcodes = 16
        len_subcodes = 16
        for i in range(num_subcodes):
            start = i * len_subcodes
            end = start + len_subcodes
            long_f += [binstr2uint(str_256_permuted[start:end])]

        codes_short = [f0, f1, f2, f3, short_r0]
        codes_long64 = [long64_f0, long64_f1, long64_f2, long64_f3, r0, r1, r2, r3]
        codes_long = [*long_f, r0, r1, r2, r3]

        csv_writer_short.writerow(prec + codes_short + succ)
        csv_writer_long.writerow(prec + codes_long + succ)
        csv_writer_long64.writerow(prec + codes_long64 + succ)

    csv_file.close()
    csv_file_long.close()
    csv_file_long64.close()
    csv_file_short.close()
