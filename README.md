# ElasticHash

Semantic Image Similarity Search in Elasticsearch

## Usage

Perform image similarity search on ~7M images of the OpenImages dataset.

* Go to app directory `cd dh`
* Download model and unpack `./get_model.sh`
* Start containers:
    - Run on CPU: `docker-compose up -d`
    - Or, if a GPU is available, run on GPU: `docker-compose --env-file ./.env.gpu up -d`
* Download OpenImages ES index and import data into ES: `docker exec dh_app /tools/import_openimages.sh`
* Go to http://localhost

## Index custom image dataset

You can also perform image similarity search on a custom image dataset. However, for reasonable results, this requires a
directory with enough images to index.

* Go to app directory `cd dh`
* Modify `docker-compose.yaml`: Add `- path/to/mage_dir/:/app/static/images/` for a path to a folder containing images
  to `app`
* Start containers `docker-compose up -d`
* Run `docker exec dh_app /tools/import_dir.sh`
* Go to http://localhost

## Credits

The demo app uses [Natural Gallery JS](https://github.com/Ecodev/natural-gallery-js)