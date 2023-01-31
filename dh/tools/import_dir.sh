#!/bin/bash
cd "$(dirname "$0")"

echo "Creating neighbors index..."
python3 /tools/create_nbs_index.py

echo "Adding hamming distance function..."
python3 /tools/add_hdist.py

echo "Importing directory..."
# Example: python3 /tools/import_dir.py --images_dir=/app/static/images --filter_prefix "2_" --allowed_files png
python3 /tools/import_dir.py --images_dir=/app/static/images
