import csv
import random
from util import es_query, parse_el_result
import json
import os
import progressbar
import requests

shuffle = True
random.seed(10)

csv_twostage = "queries/val.csv"  # Validation data
csv_long = "queries/val.long.csv"
csv_short = "queries/val.short.csv"

result_dir = "results"

num_queries = 10000  # 121588
ks = [10, 25, 50, 100, 250, 500, 1000]
num_results = max(ks)


def clear_cache(index):
    return (requests.post("http://elasticsearch:9200" + "/" + index + "/_cache / clear", verify=False).text)


def read_csv_to_dict(csv_file, filter_keys=["f0", "f1", "f2", "f3"], rerank_keys=["r0", "r1", "r2", "r3"], id_field=0,
                     code_fields_start=3):
    num_filter_keys = len(filter_keys)
    num_rerank_keys = len(rerank_keys)
    d = {}
    with open(csv_file, "r") as f:
        cr = csv.reader(f, delimiter=",")
        for row in cr:
            id = row[id_field]
            codes_filter = row[code_fields_start:code_fields_start + num_filter_keys]
            codes_rerank = row[
                           code_fields_start + num_filter_keys:code_fields_start + num_filter_keys + num_rerank_keys]
            d_rerank = {k: v for (k, v) in zip(rerank_keys, codes_rerank)}
            d_filter = {k: v for (k, v) in zip(filter_keys, codes_filter)}
            d[id] = {"f": d_filter, "r": d_rerank}
    return d


def filter(d, keys):
    return {k: v for (k, v) in d.items() if k in keys}


d_twostage = read_csv_to_dict(csv_twostage)
d_short = read_csv_to_dict(csv_short, rerank_keys=["r0"])
d_long = read_csv_to_dict(csv_long,
                          filter_keys=["f0", "f1", "f2", "f3", "f4", "f5",
                                       "f6", "f7", "f8", "f9", "f10", "f11", "f12",
                                       "f13", "f14", "f15"])
keys = list(d_twostage.keys())
if shuffle:
    random.shuffle(keys)

if len(keys) > num_queries:
    keys = keys[:num_queries]

    d_short = filter(d_short, keys)
    d_twostage = filter(d_twostage, keys)
    d_long = filter(d_long, keys)


def get_time(q_imageid, index, d_query, code_len, k):
    r = d_query[q_imageid]["r"]
    f = d_query[q_imageid]["f"]
    res = es_query(r, f, "http://elasticsearch:9200", index, profiling=True, max_res=k)
    res_j = json.loads(res)

    # print(res_j["took"])

    return res_j["took"]


csv_files = []
csv_writers = []

pb_widgets = [
    progressbar.Percentage(),
    ' ', progressbar.Counter(),
    ' ', progressbar.Bar(marker='*'),
    ' ', progressbar.ETA(),
    ' ', progressbar.FileTransferSpeed(),
]

bar = progressbar.ProgressBar(maxval=len(keys) * 3, \
                              widgets=pb_widgets)

bar.start()

indices = [
    ["es-long", 256, d_long],
    ["es-twostage", 256, d_twostage],
    ["es-short", 64, d_short]
]
i = 0
for (index, code_len, query_dict) in indices:

    for k in ks:
        f = open(os.path.join(result_dir, "times." + str(index) + "." + str(k) + ".csv"), "w")
        f.close()

    for imageid in keys:
        i += 1
        bar.update(i)
        clear_cache(index)
        clear_cache("es-nbs")
        for k in ks:
            with open(os.path.join(result_dir, "times." + str(index) + "." + str(k) + ".csv"), "a") as f:
                t = get_time(imageid, index, query_dict, code_len=code_len, k=k)
                csv_writer = csv.writer(f)
                csv_writer.writerow([imageid, t])
bar.finish()
