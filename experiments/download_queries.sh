#!/bin/bash
cd "$(dirname "$0")"

echo "Downloading queries.."
wget -O  queries.zip https://pc12439.mathematik.uni-marburg.de/nextcloud/s/zJAAZSwyqZfCYcf/download
echo "Extracting..."
unzip queries.zip -d queries  && rm queries.zip

echo "Done."

