import sys

sys.path.append('/app/')
import argparse
import progressbar
import threading
import urllib3
import requests
import queue
import time
import csv
import json

from util import str2bool

urllib3.disable_warnings()

num_files_per_request = 40
num_threads = 4

q = queue.Queue()
lock = threading.Lock()
count = 0

headers = {
    'User-Agent': 'Import Client'}

es_index_tpl_str_default = """
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

es_index_tpl_str_short = """
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
        "r0": {"type": "long"}
      }
    }
}
  """

es_index_tpl_str_long = """
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
        "f4": {"type": "keyword"},
        "f5": {"type": "keyword"},
        "f6": {"type": "keyword"},
        "f7": {"type": "keyword"},
        "f8": {"type": "keyword"},
        "f9": {"type": "keyword"},
        "f10": {"type": "keyword"},
        "f11": {"type": "keyword"},
        "f12": {"type": "keyword"},
        "f13": {"type": "keyword"},
        "f14": {"type": "keyword"},
        "f15": {"type": "keyword"},
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
    if method == "long":
        es_index_tpl_str = es_index_tpl_str_long
    elif method == "short":
        es_index_tpl_str = es_index_tpl_str_short
    else:
        es_index_tpl_str = es_index_tpl_str_default
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
    # print (r.text)


def worker():
    while True:
        batch = q.get()
        if batch is None:
            break
        process_batch(batch)
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
        description='Create ES index and import image codes from csv file')

    parser.add_argument(
        '--es_url',
        default="http://elasticsearch:9200",
        type=str,
        help='Elastic Search URL with port (default: http://elasticsearch:9200)'
    )

    parser.add_argument(
        '--csv',
        required=True,
        type=str,
        help='Path to input list containing image pathes and codes'
    )

    parser.add_argument(
        '--sep',
        default=',',
        type=str,
        help='Separator for image list (default: \',\')'
    )

    parser.add_argument(
        '--col_codes',
        default=3,
        type=int,
        help='First column id of codes (starts with 0, default: 3)'
    )

    parser.add_argument(
        '--col_imagepath',
        default=1,
        type=int,
        help='Column of imagepath (starts with 0, default: 1)'
    )

    parser.add_argument(
        '--col_imageurl',
        default=1,
        type=int,
        help='Column of imageurl (starts with 0, default: 1)'
    )

    parser.add_argument(
        '--col_thumburl',
        default=2,
        type=int,
        help='Column of thumburl (starts with 0, default: 2)'
    )

    parser.add_argument(
        '--col_id',
        default=0,
        type=int,
        help='Column with id (starts with 0, default: 0)'
    )

    parser.add_argument(
        '--col_imageinfo',
        default=11,
        type=int,
        help='Column from where image info starts. As in OpenImages there are 4 columns: license, authorprofileurl,	author,	title (starts with 0, default: 12)'
    )

    parser.add_argument('-c', '--cols',
                        choices=['id', 'imageurl', 'thumburl', 'imagepath', 'imageinfo', 'license', 'author', 'title',
                                 'authorprofileurl'], nargs="*",
                        default=['id', 'imageurl', 'thumburl'],
                        help='Columns of CSV to use for import.')

    parser.add_argument(
        '--es_reset_index',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Reset Elastic Search index before adding (default: False)'
    )

    parser.add_argument(
        '--es_index',
        default="es-retrieval",
        type=str,
        help='Elastic Search index name (default: es-retrieval)'
    )

    parser.add_argument(
        '--method',
        default="twostage",
        choices={"twostage", "short", "long"},
        type=str,
        help='Method to use (default: twostage)'
    )

    args = parser.parse_args()

    es_index = args.es_index
    es_reset_index = args.es_reset_index
    input_list = args.csv
    sep = args.sep
    col_id = args.col_id
    col_imagepath = args.col_imagepath
    col_imageurl = args.col_imageurl
    col_thumburl = args.col_thumburl
    col_codes = args.col_codes
    col_imageinfo = args.col_imageinfo
    cols = set(args.cols)
    method = args.method

    # Read CSV
    csv_file = open(input_list, "r")
    num_files_total = sum(1 for i in csv_file)
    csv_file.seek(0)
    csv_reader = csv.reader(csv_file, delimiter=sep)

    # num_files_total =

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

    # ES index
    if es_reset_index:
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
        batch = []
        for x in range(0, end - start):
            row = next(csv_reader)
            if method == "long":
                d = {
                    "f0": row[col_codes + 0],
                    "f1": row[col_codes + 1],
                    "f2": row[col_codes + 2],
                    "f3": row[col_codes + 3],
                    "f4": row[col_codes + 4],
                    "f5": row[col_codes + 5],
                    "f6": row[col_codes + 6],
                    "f7": row[col_codes + 7],
                    "f8": row[col_codes + 8],
                    "f9": row[col_codes + 9],
                    "f10": row[col_codes + 10],
                    "f11": row[col_codes + 11],
                    "f12": row[col_codes + 12],
                    "f13": row[col_codes + 13],
                    "f14": row[col_codes + 14],
                    "f15": row[col_codes + 15],
                    "r0": row[col_codes + 16],
                    "r1": row[col_codes + 17],
                    "r2": row[col_codes + 18],
                    "r3": row[col_codes + 19]
                }
            elif method == "short":
                d = {
                    "f0": row[col_codes + 0],
                    "f1": row[col_codes + 1],
                    "f2": row[col_codes + 2],
                    "f3": row[col_codes + 3],
                    "r0": row[col_codes + 4]
                }
            else:
                d = {
                    "f0": row[col_codes + 0],
                    "f1": row[col_codes + 1],
                    "f2": row[col_codes + 2],
                    "f3": row[col_codes + 3],
                    "r0": row[col_codes + 4],
                    "r1": row[col_codes + 5],
                    "r2": row[col_codes + 6],
                    "r3": row[col_codes + 7]
                }
            if "id" in cols:
                d["id"] = row[col_id]
            if "imagepath" in cols:
                d["imagepath"] = row[col_imagepath]
            if "imageurl" in cols:
                d["imageurl"] = row[col_imageurl]
            if "thumburl" in cols:
                d["thumburl"] = row[col_thumburl]
            if "imageinfo" in cols:
                d["imageinfo"] = True
                d["license"] = row[col_imageinfo]
                d["authorprofileurl"] = row[col_imageinfo + 1]
                d["author"] = row[col_imageinfo + 2]
                d["title"] = row[col_imageinfo + 3]
            else:
                d["imageinfo"] = False

            batch += [d]

        q.put(batch)
    q.join()

    # stop workers
    for i in range(num_threads):
        q.put(None)
    for t in threads:
        t.join()

    bar.finish()

    csv_file.close()

    duration = time.time() - s
    print("""
    -------------------------------------------------------

    Total time: %0.2fs for %d images
    Time per image: %0.2fs
    """ % (duration, num_files_total, duration / num_files_total))
