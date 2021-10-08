import os


def get_env(env, default=""):
    env_val = os.environ.get(env)
    if env_val:
        return env_val
    else:
        return default


class Config:
    # Web App settings
    MAX_UPLOAD_SIZE = 5000  # kB
    ALLOWED_EXT = ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"]
    IMAGES_URL = get_env("IMAGES_URL", "")
    THUMBS_URL = get_env("THUMBS_URL", "")
    LOCAL_IMAGES = get_env("LOCAL_IMAGES", False)

    # Elastic Search settings
    ES_MAX_RESULTS = 1000
    ES_URL = get_env("ES_URL", "http://elasticsearch:9200/es-retrieval/_search?size=")

    with open("/app/templates/es_query.7.x.json", 'r') as file:
        EL_QUERY_TPL = file.read()

    # Image settings
    IMG_WIDTH = 300
    IMG_HEIGHT = 300

    # Inference
    TF_SERVING_HOST = get_env("TF_SERVING_HOST", "tf-serving")
    TF_SERVING_PORT = int(get_env("TF_SERVING_PORT", "8501"))
    TF_MODELNAME = get_env("TF_MODELNAME", "model")


import logging

logger = logging.Logger("logs")
