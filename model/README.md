# Train

# Prepare model

Prepare a trained Keras model for ElasticHash. 

Model requirements:

* Must contain a coding layer with 256 units
* Name of coding layer: "code"

Perform the following steps to first use an intermediate model to compute 256 bit codes. 
The codes are then decorrelated and 64 bit and a 256 bit layer added.

* Specify the following settings
    ```  
    # Directory with images in subdirs
    IMAGE_DIR=/path/to/image_dir/
  
    # Path to the trained Keras model (*.keras)
    TRAINED_MODEL=/path/to/model.keras
  
    GPU_IDS=0
  
    IMG_WIDTH=450
    IMG_HEIGHT=150
    ```
  
* Create container for exporting the models
    ``` 
    docker build -t export_model .
    ```

* Export intermediate model
    ```  
    nvidia-docker run -it -e IMG_WIDTH=${IMG_WIDTH} -e IMG_HEIGHT=${IMG_HEIGHT} \
    -e CUDA_VISIBLE_DEVICES=${GPU_IDS}  -v $(pwd)/prepare_model:/code/ -v ${TRAINED_MODEL}:/trained_model \
    export_model \
    python /code/export_model.py --keras_weights /trained_model --output_dir /code/intermediate_model
    ```
  
* Start tf-serving container with intermediate model
    ``` 
    INTERMEDIATE_MODEL_DIR="$(pwd)/prepare_model/intermediate_model/model"
    nvidia-docker run -d -p 8501:8501 -e CUDA_VISIBLE_DEVICES=${GPU_IDS} -e MODEL_NAME=model  \
    -e IMG_WIDTH=${IMG_WIDTH} -e IMG_HEIGHT=${IMG_HEIGHT}  -v $(pwd)/prepare_model:/code/ \
    -v ${INTERMEDIATE_MODEL_DIR}/$(ls -t ${INTERMEDIATE_MODEL_DIR} | head -1):/models/model/1 --network "host" \
    --name intermediate_model tensorflow/serving:2.4.1-gpu
    ```
  
* Compute codes with the intermediate model and decorrelate bits

  * Generate a list of all images in subdirs
    ``` 
    nvidia-docker run -it -e CUDA_VISIBLE_DEVICES=${GPU_IDS}  -v $(pwd)/prepare_model:/code/ -v ${IMAGE_DIR}:/images/\
    export_model \
    python /code/generate_csv.py --images_dir /images/ --allowed_files png --output /code/images.csv --filter_prefix "2_"
    ```
  * Compute codes for images
    ``` 
    nvidia-docker run -it -e CUDA_VISIBLE_DEVICES=${GPU_IDS}  -e TF_SERVING_HOST=127.0.0.1 \
    -e IMG_WIDTH=${IMG_WIDTH} -e IMG_HEIGHT=${IMG_HEIGHT}  \
    -v $(pwd)/prepare_model:/code/ -v ${IMAGE_DIR}:/images/ --network "host" \
    export_model \
    python /code/inference_csv.py  --image_dir /images  --input_list /code/images.csv \
    --output_list /code/images.with.codes.csv
    ```
  * Compute correlations and decorrelate
    ``` 
    shuf prepare_model/images.with.codes.csv > prepare_model/images.with.codes.shuf.csv
    
    nvidia-docker run -it -v $(pwd)/prepare_model:/code/ export_model \
    python /code/correlations.py  --num_codes 50000 --input_list /code/images.with.codes.shuf.csv --output_dir /code/
    
    nvidia-docker run -it -v $(pwd)/prepare_model:/code/ export_model \
    python /code/decorrelate.py  --corr_file /code/256_corr.txt --output_dir /code/
    ```

* Export final model
  ```  
  nvidia-docker run -it -e CUDA_VISIBLE_DEVICES=${GPU_IDS}  -v $(pwd)/prepare_model:/code/ -v ${TRAINED_MODEL}:/trained_model \
  -e IMG_WIDTH=${IMG_WIDTH} -e IMG_HEIGHT=${IMG_HEIGHT}  export_model \
  python /code/export_model.py --keras_weights /trained_model --output_dir /code/final_model --split_and_permute \
  --permute_64 /code/64_16_from_256_perm.pkl --permute_256 /code/256_16_perm.pkl
  ```
  
* Serve final model
  ``` 
  FINAL_MODEL_DIR="$(pwd)/prepare_model/final_model/model"
  
  nvidia-docker stop intermediate_model
  
  nvidia-docker run -d -p 8501:8501 -e CUDA_VISIBLE_DEVICES=${GPU_IDS} -e MODEL_NAME=model -v $(pwd)/prepare_model:/code/ \
  -v ${FINAL_MODEL_DIR}/$(ls -t ${FINAL_MODEL_DIR} | head -1):/models/model/1 --network "host" \
  --name final_model tensorflow/serving:2.4.1-gpu
 
  ```
*  Finally you can put the exported model to `dh/models/` and use it for similarity search (to index your data follow the steps to [index a custom image dataset](../README.md))

