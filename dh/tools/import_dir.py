import sys

sys.path.append('/app/')
import os
import argparse
import progressbar
import threading
import json
import urllib3
import requests
import queue
import time
from util import allowed_file, str2bool, batch_inference, load_image
from urllib.parse import quote

urllib3.disable_warnings()

es_index = "es-retrieval"

num_files_per_request = 10
num_threads = 4

q = queue.Queue()
lock = threading.Lock()
count = 0

headers = {
    'User-Agent': 'Import Client'}

es_index_tpl_str = """
{
  "settings": {
    "number_of_shards": 3
  },
  "mappings": {
      "properties": {
        %s
        "f0": {"type": "keyword"},
        "f1": {"type": "keyword"},
        "f2": {"type": "keyword"},
        "f3": {"type": "keyword"},
        "r0": {"type": "long"},
        "r1": {"type": "long"},
        "r2": {"type": "long"},
        "r3": {"type": "long"}
      }
    }
}
  """

pb_widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]


def es_drop_and_create_index():
    # Drop index
    r = requests.delete(args.es_url + "/" + es_index, verify=False)
    print(args.es_url, r.text)
    # Create index
    cols = ['id', 'imageurl', 'thumburl']
    data_fields = "".join(
        [" \"" + field_name + "\": { \"type\": \"keyword\", \"index\": false }, " for field_name in cols])
    q = es_index_tpl_str % (data_fields,)
    print(q)
    r = requests.put(args.es_url + "/" + es_index, q, headers={'Content-Type': 'application/json'})
    print(r.text)
    # No read-only
    r = requests.put(args.es_url + "/" + es_index + "/_settings",
                     """{"index": {"blocks": {"read_only_allow_delete": "false"}}}""",
                     headers={'Content-Type': 'application/json'})
    print(r.text)


def process_batch(batch):
    s = ""
    for r in batch:
        s += """{ "index": { "_index":"%s" } }
        """ % (es_index,)
        s += json.dumps(r).replace('\n', ' ') + "\n"
    # print (s)
    r = requests.post(args.es_url + "/" + es_index + "/_bulk", s, headers={"Content-Type": "application/x-ndjson"})
    print(r.text)


def process_batch(batch):
    results = batch_inference(batch)
    # results = json.loads(r.text)["results"]
    s = ""
    for r in results:
        code_dict = r["codes"]
        webpath = os.path.join("static/images/",quote(r["fieldname"]))
        code_dict["imageurl"] = webpath
        code_dict["thumburl"]= webpath
        code_dict["id"] = r["fieldname"]
        s += """{ "index": { "_index":"%s" } }
        """ % (es_index,)
        s += json.dumps(code_dict).replace('\n', ' ') + "\n"
        # s += es_generate_doc_str(code_dict).replace('\n', ' ') + "\n"
    r = requests.post(args.es_url + "/" + es_index + "/_bulk", s, headers={"Content-Type": "application/x-ndjson"})

def get_image_batch(batch):
    files = {}
    for p in batch:
        local_file = open(os.path.join(image_dir, p), "rb")
        img = local_file.read()
        img = load_image(img)
        files[p] = {p: img}
    return files


def worker():
    while True:
        batch = q.get()
        if batch is None:
            break
        image_batch = get_image_batch(batch)
        process_batch(image_batch)
        global count
        global bar
        lock.acquire()
        count += num_files_per_request
        count = count if count <= num_files_total else num_files_total
        bar.update(count)
        lock.release()
        q.task_done()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create ES index, compute and import image from imagefiles')

    parser.add_argument(
        '--es_url',
        default="http://elasticsearch:9200",
        type=str,
        help='Elastic Search URL with port (default: http://elasticsearch:9200)'
    )

    parser.add_argument(
        '--es_index',
        default="es-retrieval",
        type=str,
        help='Elastic Search index name (default: es-retrieval)'
    )

    parser.add_argument(
        '--images_dir',
        default="images",
        type=str,
        help='Directory containing keyframes'
    )

    args = parser.parse_args()

    image_dir = args.images_dir
    es_index = args.es_index

    pathes = []

    print("""Reading images...""")

    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if (allowed_file(filename)):
                path = os.path.join(os.path.relpath(root, image_dir), filename)
                print ("Adding " + path)
                pathes.append(path)

    #    pathes = pathes[:10000]

    num_files_total = len(pathes)
    print("""%d images found.""" % (num_files_total,))

    # ES index
    if not es_index.islower():
        raise ("Index needs to be lowercase")

    # Init threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    es_drop_and_create_index()

    s = time.time()
    threads = []
    print("Processing images...")

    bar = progressbar.ProgressBar(maxval=num_files_total, \
                                  widgets=pb_widgets)

    bar.start()

    for start in range(0, num_files_total, num_files_per_request):
        end = start + num_files_per_request
        end = end if end <= num_files_total else num_files_total
        batch = pathes[start:end]
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
