#!/bin/bash
cd "$(dirname "$0")"

echo "Downloading ElasticSearch index..."
wget -O  esindex.tar.gz https://data.uni-marburg.de/bitstream/handle/dataumr/233/esdata-experiments.tar.gz
echo "Extracting..."
tar -xf esindex.tar.gz --directory ./es-data && rm esindex.tar.gz
chown -R 1000:1000 es-data
echo "Done."

