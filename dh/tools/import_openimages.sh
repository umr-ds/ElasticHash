#!/bin/bash
cd "$(dirname "$0")"

echo "Downloading codes..."
wget -O  codes.tar.gz https://pc12439.mathematik.uni-marburg.de/nextcloud/s/APKg3gLT5Q9rPko/download
echo "Extracting codes..."
tar -xf codes.tar.gz
rm codes.tar.gz

echo "Creating neighbors index..."
python3 create_nbs_index.py

echo "Adding hamming distance function..."
python3 add_hdist.py

echo "Importing OpenImages codes..."
python3 import_csv.py --es_reset_index --csv train.csv
# python3 import_csv.py --csv val.csv

