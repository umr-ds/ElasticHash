# Experiments

Running experiments requires some disk space: ~10G for ElasticSearch indices and ~75G for OpenImages annotations in
Postgres DB

## Initialize

* Prepare Elastic Search: Download and extract index: `./download_esindex.sh`

* Prepare OpenImages Postgres DB
    * Download OpenImages labels `sh ./download_annotations.sh`
    * Start docker `docker-compose up -d`
    * Import OpenImages labels `docker exec -it -u postgres dh_db psql -d openimages -a -f /import.sql`
      Note: This may take a long time.

* Download queries: `sh ./download_queries.sh`

* Scripts for reproducing the experiments:
    * `get_aps.py`
    * `get_times.py`

## Misc

* Generate OpenImages CSVs for experiments:
    * `python3 generate_long_short.py --csv /tools/train.csv`
    * `python3 generate_long_short.py --csv /tools/val.csv`

* Import CSV into ElasticSearch
    * 2-stage `python3 import_csv.py --es_index es-twostage --es_reset_index --csv /tools/train.csv --cols id`
    * 64-bit `python3 import_csv.py --es_index es-short --es_reset_index --csv /tools/train.short.csv --cols id`
    *
    256-bit `python3 import_csv.py --es_index es-long --es_reset_index --csv /tools/train.long.csv --cols id --method long`