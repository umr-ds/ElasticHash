#!/bin/bash
cd "$(dirname "$0")"

echo "Downloading queries.."
wget -O  queries.zip https://hessenbox.uni-marburg.de/dl/fiBUXEoHiizwzxLv5HGRnY/val_queries.zip
echo "Extracting..."
unzip queries.zip -d queries  && rm queries.zip

echo "Done."

