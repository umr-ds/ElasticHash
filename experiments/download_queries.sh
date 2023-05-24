#!/bin/bash
cd "$(dirname "$0")"

echo "Downloading queries.."
wget -O  queries.zip https://data.uni-marburg.de/bitstream/handle/dataumr/233/val_queries.zip
echo "Extracting..."
unzip queries.zip -d queries  && rm queries.zip

echo "Done."

