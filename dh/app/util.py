import argparse
import base64
import io
import json
from io import BytesIO
from string import Template

import numpy as np
import requests
from PIL import Image as pil_image
from bitstring import BitArray

from config import Config as cfg


def get_hashcode_str(floatarr):
    """
    Construct hashcode from layer output (np.array)
    :param floatarr: np.array
    :return: string
    """
    code = (floatarr > 0).astype(np.int)
    s = "".join(map(str, code))
    return s


def tf_inference(imgs_b64, protocol, host, port, modelname):
    """
    Send batch of base64 images to tf-serving rest api and return resuls
    :param imgs_b64: array of base64 encoded images
    :param protocol: http or https
    :param host: tf-serving hostname
    :param port: tf-serving port
    :param modelname: name of served model
    :return: predictions returned by tf-serving
    """
    data = json.dumps({"signature_name": "serving_default", "instances": imgs_b64})
    res = requests.post('%s://%s:%d/v1/models/%s:predict' % (protocol, host, port, modelname), data=data,
                        headers={"content-type": "application/json"})
    json_res = json.loads(res.text)
    return np.array(json_res['predictions'])


def load_image(imagedata):
    """
    Read and convert image with PIL
    :param imagedata: raw image data
    :return: PIL Image
    """
    buffer = BytesIO()
    buffer.write(imagedata)
    buffer.seek(0)
    image = pil_image.open(buffer)
    # Convert to RGB
    if image.mode in ['RGB']:
        # No conversion necessary
        pass
    elif image.mode in ['1']:
        # Easy conversion to L
        image = image.convert('L').convert('RGB')
    elif image.mode in ['LA']:
        # Deal with transparencies
        new = pil_image.new('L', image.size, 255)
        new.paste(image, mask=image.convert('RGBA'))
        image = new
        image.convert('RGB')
    elif image.mode in ['CMYK', 'YCbCr', 'L']:
        # Easy conversion to RGB
        image = image.convert('RGB')
    elif image.mode in ['P', 'RGBA']:
        # Deal with transparencies
        new = pil_image.new('RGB', image.size, (255, 255, 255))
        new.paste(image, mask=image.convert('RGBA'))
        image = new
    else:
        raise Exception('Image mode "%s" not supported' % image.mode)
    return image


def binstr2int(s):
    """
    Convert binary string to signed integer
    :param s: string
    :return: signed int
    """
    b = BitArray(bin=s)
    return b.int


def binstr2uint(s):
    """
    Convert binary string to unsigned integer
    :param s: string
    :return: unsigned int
    """
    b = BitArray(bin=s)
    return b.uint


def resize_and_crop(image, new_width, new_height, crop=True):
    """
    Resize image and maintain aspect ration, then crop
    :param image: input image as PIL Image
    :param new_width: width of crop
    :param new_height: height of crop
    :param crop: False if only resizing is required
    :return: resized (and cropped) image as PIL Image
    """
    width, height = image.size
    width_ratio = float(width) / new_width
    height_ratio = float(height) / new_height
    # Resize to smallest of ratios (relatively larger image), keeping aspect ratio
    if width_ratio > height_ratio:
        resize_width = int(round(width / height_ratio))
        resize_height = new_height
    else:
        resize_height = int(round(height / width_ratio))
        resize_width = new_width

    # image = np.array(image.resize((resize_width, resize_height), resample=pil_image.BILINEAR))
    image = image.resize((resize_width, resize_height), resample=pil_image.BILINEAR)

    if crop:
        if width_ratio > height_ratio:
            start = int(round((resize_width - new_width) / 2.0))
            # image = image[:, start:start + new_width]
            image = image.crop((start, 0, start + new_width, new_height))
        else:
            start = int(round((resize_height - new_height) / 2.0))
            # image = image[start:start + new_height, :]
            # left, upper, right, and lower pixel
            image = image.crop((0, start, new_width, start + new_height))
    return image


def allowed_file(filename, allowed_ext=cfg.ALLOWED_EXT):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_ext


def parse_es_results(json_str):
    """
    Parse ES results to image dicts (imagepath, score, id)
    :param json_str: JSON formatted string
    :return: list of dicts
    """
    json_result = json.loads(json_str)
    hits = json_result["hits"]["hits"]
    imgs = []
    for id, h in enumerate(hits):
        d = {"imageid": h["_id"], "id": id, "score": h["_score"]}
        if "imagepath" in h["_source"]:
            d["imagepath"] = h["_source"]["imagepath"]
            d["thumbpath"] = h["_source"]["imagepath"]
        elif "imageurl" in h["_source"]:
            d["imagepath"] = h["_source"]["imageurl"]
            if "thumburl" in h["_source"]:
                d["thumbpath"] = h["_source"]["thumburl"]
            else:
                d["thumbpath"] = h["_source"]["imageurl"]
        if "imageinfo" in h["_source"]:
            d["imageinfo"] = {
                "license": h["_source"]["license"],
                "authorprofileurl": h["_source"]["authorprofileurl"],
                "author": h["_source"]["author"],
                "title": h["_source"]["title"],
            }
        imgs.append(d)
    return imgs


def batch_inference(files):
    """
    Perform inference with tf-serving for all images in batch of image files.
    :param files: { field_name : { file_name : buffered image } }
    :return:
    """
    response = []

    # Read files to base64 strings
    imgs_b64 = []
    filenames = []
    filekeys = []
    for key, fd in files.items():
        filename, img = list(fd.items())[0]
        # buf = file.read()
        # img = load_image(buf)
        # logger.error(img)
        img = resize_and_crop(img, cfg.IMG_WIDTH, cfg.IMG_HEIGHT)
        ba = io.BytesIO()
        img.save(ba, format='BMP')
        imgs_b64 += [{"b64": base64.b64encode(ba.getvalue()).decode('utf-8')}]
        filenames += [filename]
        filekeys += [key]
        ba.close()

    # Process with tf serving
    preds = tf_inference(imgs_b64, "http", cfg.TF_SERVING_HOST, cfg.TF_SERVING_PORT, cfg.TF_MODELNAME)
    for pred, key, filename in zip(preds, filekeys, filenames):
        code64_str = get_hashcode_str(np.array(pred["code64"], dtype=float))
        code256_str = get_hashcode_str(np.array(pred["code256"], dtype=float))
        res = {
            'codes': {
                'f0': str(binstr2uint(code64_str[:16])),
                'f1': str(binstr2uint(code64_str[16:32])),
                'f2': str(binstr2uint(code64_str[32:48])),
                'f3': str(binstr2uint(code64_str[48:])),
                'r0': binstr2int(code256_str[:64]),
                'r1': binstr2int(code256_str[64:128]),
                'r2': binstr2int(code256_str[128:192]),
                'r3': binstr2int(code256_str[192:])

            },
            'filename': filename,
            'fieldname': key,
            "status_code": "OK"
        }
        response.append(res)
    return response


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def es_query_str(code_dict):
    s = Template(cfg.EL_QUERY_TPL)
    return s.substitute(**code_dict)


def es_query(query, max_results):
    return requests.post(cfg.ES_URL + str(min(cfg.ES_MAX_RESULTS, max_results)), data=query,
                         headers={"Content-Type": "application/json"}, verify=False).text


def monkeypatch_imghdr():
    """
    Monkey patch bug in imghdr which causes valid JPEG images to be classified as 'None' type.
    Return additional testing methods to patch the current imghdr instance."""
    return test_small_header_jpeg, test_exif_jfif


def test_small_header_jpeg(h, f):
    """JPEG data with a small header"""
    jpeg_bytecode = b'\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06' \
                    b'\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f'
    if len(h) >= 32 and h[5] == 67 and h[:32] == jpeg_bytecode:
        return 'jpeg'


def test_exif_jfif(h, f):
    """JPEG data in JFIF or Exif format"""
    if h[6:10] in (b'JFIF', b'Exif') or b'JFIF' in h[:23] or h[:2] == b'\xff\xd8':
        return 'jpeg'
