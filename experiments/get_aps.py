import csv
import random
from util import es_query, parse_el_result
import json
from openimages import OpenImagesDB
from sklearn.metrics import average_precision_score as ap
import os
import progressbar

shuffle = False
random.seed(10)

csv_twostage = "queries/val.csv"  # Validation data
csv_long = "queries/val.long.csv"
csv_short = "queries/val.short.csv"

result_dir = "results"

confidence = 0.5

num_queries = 121588
ks = [10, 25, 50, 100, 150, 200, 250, 300, 500]
num_results = max(ks)

db = OpenImagesDB(database="openimages", user="postgres", host="db", password="oi_secure_pw",
                  port=5432)


def common_label(a, b):
    return int(len(set(a).intersection(set(b))) > 0)


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


def get_gt(q_imageid, index, d_query, code_len):
    query_labels = db.get_image_labels(q_imageid)
    r = d_query[q_imageid]["r"]
    f = d_query[q_imageid]["f"]
    res = es_query(r, f, "http://elasticsearch:9200", index, profiling=False, max_res=num_results)
    res_j = json.loads(res)

    ret = parse_el_result(res_j)

    gt = []
    probs = []

    for rs in ret["hits"]:
        probs += [rs["score"] / code_len]
        imageid = rs["id"]
        labels = db.get_image_labels(imageid, confidence=confidence)
        gt.append(common_label(labels, query_labels))
    return probs, gt


csv_files = []
csv_writers = []
for k in ks:
    f = open(os.path.join(result_dir, "aps." + str(k) + ".csv"), "w")
    f.close()

pb_widgets = [
    progressbar.Percentage(),
    ' ', progressbar.Counter(),
    ' ', progressbar.Bar(marker='*'),
    ' ', progressbar.ETA(),
    ' ', progressbar.FileTransferSpeed(),
]

bar = progressbar.ProgressBar(maxval=len(keys), \
                              widgets=pb_widgets)

bar.start()

for i, imageid in enumerate(keys):
    bar.update(i)
    p_short, gt_short = get_gt(imageid, "es-short", d_short, code_len=64)
    p_twostage, gt_twostage = get_gt(imageid, "es-twostage", d_twostage, code_len=256)
    p_long, gt_long = get_gt(imageid, "es-long", d_long, code_len=256)
    for k in ks:
        with open(os.path.join(result_dir, "aps." + str(k) + ".csv"), "a") as f:
            ap_short = ap(gt_short[:k], p_short[:k]) if 1 in gt_short[:k] else 0
            ap_twostage = ap(gt_twostage[:k], p_twostage[:k]) if 1 in gt_twostage[:k] else 0
            ap_long = ap(gt_long[:k], p_long[:k]) if 1 in gt_long[:k] else 0
            csv_writer = csv.writer(f)
            csv_writer.writerow([imageid, ap_short, ap_twostage, ap_long])
bar.finish()
