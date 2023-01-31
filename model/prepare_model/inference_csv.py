import sys
import os
import time
import threading
import argparse
import traceback
import urllib3
import progressbar
import csv
import queue
# from settings import *
from config import Config as cfg
import numpy as np
from util import load_image, resize_and_crop, tf_inference
import io
import base64


def get_hashcode_string(floatarr):
    """
    Construct hashcode from layer output (np array)
    :param floatarr: np.array
    :return: string
    """
    code = (floatarr > 0).astype(np.int)
    s = "".join(map(str, code))
    return s


urllib3.disable_warnings()

q = queue.Queue()

lock_outfile = threading.Lock()
lock_errorfile = threading.Lock()
count = 0
pb_widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]

dim = (cfg.IMG_HEIGHT, cfg.IMG_WIDTH)


def process_batch_tf(batch):
    valid_rows = []
    imgs_b64 = []
    for row in batch:
        img_path = row[col]
        try:
            local_file = open(os.path.join(image_dir, img_path), "rb")
            img = local_file.read()
            img = load_image(img)
            img = resize_and_crop(img, cfg.IMG_WIDTH, cfg.IMG_HEIGHT)
            ba = io.BytesIO()
            img.save(ba, format='BMP')
            imgs_b64 += [{"b64": base64.b64encode(ba.getvalue()).decode('utf-8')}]
            ba.close()
            valid_rows += [row]
        except:
            lock_errorfile.acquire()
            with open(error_list, mode='a') as f:
                # csv_writer = csv.writer(f, delimiter=sep, quoting=csv.QUOTE_NONE)
                # csv_writer.writerow(row + [ traceback.format_exc() ])
                f.write(traceback.format_exc() + "\n")
            with open(failed_list, mode='a') as f:
                csv_writer = csv.writer(f, delimiter=sep, quoting=csv.QUOTE_NONE)
                csv_writer.writerow(row)
            lock_errorfile.release()

    # Batch inference
    rows_with_codes = []
    try:
        codes = tf_inference(imgs_b64, protocol, cfg.TF_SERVING_HOST, cfg.TF_SERVING_PORT, cfg.TF_MODELNAME)
        for row, code in zip(valid_rows, codes):
            rows_with_codes.append(row + [get_hashcode_string(code)])
    except:
        lock_errorfile.acquire()
        with open(error_list, mode='a') as f:
            f.write(traceback.format_exc() + "\n")
        lock_errorfile.release()

    return rows_with_codes


def worker():
    while True:
        batch = q.get()
        if batch is None:
            break
        new_rows = process_batch_method(batch)
        global count
        global bar
        lock_outfile.acquire()
        count += num_files_per_request
        count = count if count <= num_files_total else num_files_total
        bar.update(count)
        with open(output_list, mode='a') as f:
            csv_writer = csv.writer(f, delimiter=sep, quoting=csv.QUOTE_NONE)
            for row in new_rows:
                csv_writer.writerow(row)
        lock_outfile.release()
        q.task_done()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process a list of images (csv) and return the list with a hashcode appended to each entry.')
    parser.add_argument(
        '--num_threads',
        default=4,
        type=int,
        help='Number of threads (default: 5)'
    )
    parser.add_argument(
        '--num_files_per_request',
        default=40,
        type=int,
        help='Number of files (images) within one POST request (default: 40)'
    )

    parser.add_argument(
        '--input_list',
        required=True,
        type=str,
        help='Path to input list containing image pathes'
    )
    parser.add_argument(
        '--output_list',
        type=str,
        required=True,
        help='Path to output list'
    )
    parser.add_argument(
        '--image_dir',
        required=True,
        type=str,
        help='Directory with images (prefix for images in image_list)'
    )
    parser.add_argument(
        '--sep',
        default=',',
        type=str,
        help='Separator for image list (default: \',\')'
    )
    parser.add_argument(
        '--col',
        default=0,
        type=int,
        help='Column id of image path (default: 0)'
    )
    parser.add_argument(
        '--https',
        default=False,
        type=bool,
        help='Use https (default: False)'
    )

    args = parser.parse_args()

    num_threads = args.num_threads
    num_files_per_request = args.num_files_per_request
    host = cfg.TF_SERVING_HOST
    port = cfg.TF_SERVING_PORT
    https = args.https
    input_list = args.input_list
    output_list = args.output_list
    error_list = output_list + ".errors"
    failed_list = output_list + ".failed"
    image_dir = args.image_dir
    sep = args.sep
    col = args.col
    process_batch_method = process_batch_tf

    if https:
        protocol = "https"
    else:
        protocol = "http"

    # Init threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    # Read CSV
    data = []
    with open(input_list) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=sep)
        for row in csv_reader:
            data.append(row)

    num_files_total = len(data)

    print("""
-------------------------------------------------------
  Host:                 %s://%s:%d
  Files:                %d
  Threads:              %d
  Files per request:    %d 
-------------------------------------------------------
""" % (protocol, host, port, num_files_total, num_threads, num_files_per_request))

    s = time.time()
    threads = []
    print("Processing images...")

    with open(output_list, mode='w') as f:
        f.write("")

    with open(error_list, mode='w') as f:
        f.write("")

    with open(failed_list, mode='w') as f:
        f.write("")

    bar = progressbar.ProgressBar(maxval=num_files_total, \
                                  widgets=pb_widgets)

    bar.start()

    for start in range(0, num_files_total, num_files_per_request):
        end = start + num_files_per_request
        end = end if end <= num_files_total else num_files_total
        batch = data[start:end]
        q.put(batch)
    q.join()

    # stop workers
    for i in range(num_threads):
        q.put(None)
    for t in threads:
        t.join()

    bar.finish()

    duration = time.time() - s
    print("""
-------------------------------------------------------

Total time: %0.2fs for %d images
Time per image: %0.2fs
""" % (duration, num_files_total, duration / num_files_total))
