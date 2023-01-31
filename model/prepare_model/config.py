import os
import logging


def get_env(env, default=""):
    env_val = os.environ.get(env)
    if env_val:
        return env_val
    else:
        return default


class Config:
    ALLOWED_EXT = ["png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"]

    # Image settings
    IMG_WIDTH = int(get_env("IMG_WIDTH", 300))
    IMG_HEIGHT = int(get_env("IMG_HEIGHT", 300))

    # Inference
    TF_SERVING_HOST = get_env("TF_SERVING_HOST", "tf-serving")
    TF_SERVING_PORT = int(get_env("TF_SERVING_PORT", "8501"))
    TF_MODELNAME = get_env("TF_MODELNAME", "model")


logger = logging.Logger("logs")
