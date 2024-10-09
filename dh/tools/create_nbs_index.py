import time
import requests
import argparse
import urllib3
import progressbar
from string import Template
import numpy as np
import json
from itertools import combinations
from bitstring import BitArray

urllib3.disable_warnings()

pb_widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]

es_index_tpl_str = """{
  "settings": {
    "number_of_shards": 5
  },
  "mappings": {
    "data": {
      "properties": $properties
    }
  }
}
"""

def binstr2uint(s):
    """
    Convert binary string to unsigned integer
    :param s: string
    :return: unsigned int
    """
    b = BitArray(bin=s)
    return b.uint

def gen_masks(l=16, d=2):
    """
    Generate mask for binary strings of length l
    :param l: length of binary string
    :param d: hamming distance
    :return: list of masks as binary strings
    """
    combs = []
    ids = range(l)
    for r in range(1, d + 1):
        combs += list(combinations(ids, r))
    masks = np.zeros((len(combs), l), dtype=int)
    for i, c in enumerate(combs):
        masks[i, c] = 1
    masks_str = [(np.uint16)(binstr2uint("0" * l))] + [(np.uint16)(binstr2uint("".join(m))) for m in masks.astype(str)]
    return masks_str


def get_nbs(q, masks):
    """
    Compute neighbors by applying masks to query
    :param q: query string
    :param masks: list of binary strings
    :return: list of neighbors as binary strings
    """
    return np.bitwise_xor(q, masks, dtype=int)

def es_drop_and_create_index():
    # Drop index
    requests.delete(es_url + "/" + es_index, verify=False)
    # Create index
    s = Template(es_index_tpl_str)
    s = s.substitute(properties=json.dumps(el_fields))
    requests.put(es_url + "/" + es_index, s, headers={'Content-Type': 'application/json'})
    # No read-only
    requests.put(es_url + "/" + es_index + "/_settings",
                 """{"index": {"blocks": {"read_only_allow_delete": "false"}}}""",
                 headers={'Content-Type': 'application/json'})


def es_add_batch_to_index(batch):
    s = ""
    for id in batch:
        code_dict = {"nbs": get_nbs(id, masks).tolist()} 
        s += """{ "index": { "_id":"%s", "_index":"%s" } }
    """ % (id, es_index,) # , "_type" : "data"
        s += json.dumps(code_dict).replace('\n', ' ') + "\n"  # Needs to be 1 line for ES bulk api!
    r = requests.post(es_url + "/" + es_index + "/_bulk", s, headers={"Content-Type": "application/x-ndjson"})
    jr = json.loads(r.text)
    if "error" in jr:
        print (jr)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create ES index with all possible d1 and d2 neighbors for 16 bit codes')

    parser.add_argument(
        '--es_url',
        default="http://elasticsearch:9200",
        type=str,
        help='Elastic Search URL with port (default: http://elasticsearch:9200)'
    )

    args = parser.parse_args()

    # Try: int
    el_fields = {
        "nbs": {"type": "keyword"}, #, "eager_global_ordinals": True, "index_options:": "offsets"},
    }

    sc_len = 16
    d = 2

    num_ids = 2 ** sc_len

    ids = range(num_ids)

    es_index = "nbs"
    es_url = args.es_url

    es_drop_and_create_index()

    print("""
-------------------------------------------------------
  Entries:                %d
-------------------------------------------------------
""" % (num_ids,))

    num_lines_per_request = 500

    masks = gen_masks(sc_len, d)
    s = time.time()

    print("Processing ids...")

    bar = progressbar.ProgressBar(maxval=num_ids, \
                                  widgets=pb_widgets)

    bar.start()

    for start in range(0, num_ids, num_lines_per_request):
        bar.update(start)
        end = start + num_lines_per_request
        end = end if end <= num_ids else num_ids
        batch = ids[start:end]
        es_add_batch_to_index(batch)

    bar.finish()

    duration = time.time() - s
    print("""
-------------------------------------------------------

Total time: %0.2fs for %d entries
Time per entry: %0.4fs
""" % (duration, num_ids, duration / num_ids))

