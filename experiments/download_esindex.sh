#!/bin/bash
cd "$(dirname "$0")"

echo "Downloading ElasticSearch index..."
wget -O  esindex.tar.gz https://pc12439.mathematik.uni-marburg.de/nextcloud/s/4N9LxZJTGJJtKFA/download
echo "Extracting..."
tar -xf esindex.tar.gz --directory ./es-data && rm esindex.tar.gz
chown -R 1000:1000 es-data
echo "Done."

