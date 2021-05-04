import argparse
from itertools import combinations
import numpy as np
from bitstring import BitArray
import requests


def binstr2int(s):
    """
    Convert binary string to signed integer
    :param s: string
    :return: signed int
    """
    b = BitArray(bin=s)
    return b.int


def int2binstr(s, length=16):
    """
    Convert signed integer  to binary string
    :param s: string
    :return: signed int
    """
    b = BitArray(int=s, length=length)
    return str(b.bin)


def binstr2uint(s):
    """
    Convert binary string to unsigned integer
    :param s: string
    :return: unsigned int
    """
    b = BitArray(bin=s)
    return b.uint


def uint2binstr(s, length=16):
    """
    Convert unsigned integer  to binary string
    :param s: string
    :return: signed int
    """
    b = BitArray(uint=s, length=length)
    return str(b.bin)


def gen_masks(l=8, d=2):
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
    masks = np.zeros((len(combs), l), dtype=np.int)
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


def permute_code(bitstring, permutation):
    return "".join([bitstring[i] for i in permutation])


def reorder_code(bitstring, weights, order="desc"):
    """
    Reorders bitcode by sorting subcodes according to weights
    :param bitstring: str
    :param weights: list of weights (one for each subcode)
    :param order: asc or desc
    :return: reordered bitstring
    """
    num_sc = len(weights)
    if (len(bitstring) % num_sc != 0):
        raise ("Error: Bitstring must be divideable by number of weights")
    len_sc = len(bitstring) // num_sc
    _, order = zip(*sorted(zip(weights, range(num_sc)), reverse=(order == "desc")))
    new_bitstring = ""
    for i in range(len(bitstring)):
        new_bitstring += bitstring[(i % len_sc) + order[i // len_sc] * len_sc]
    return new_bitstring


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_el_result(j):
    out = {}
    # out["query_id"] = query_id
    # out["took"] = j["took"]
    out["max_score"] = j["hits"]["max_score"]
    out["total"] = j["hits"]["total"]["value"]
    hits = []
    for h in j["hits"]["hits"]:
        hits += [{"id": h["_source"]["id"], "score": h["_score"]}]  # _id
    out["hits"] = hits
    return out


def es_query(r, f, es_url, es_index, profiling=False, max_res=1000):
    s = '{ '
    s += '"_source": ["id"],'
    if (profiling):
        s += '"profile": true,'
    s += '"query":{"function_score":{"query":{'
    s += '"constant_score":{"boost":0,"filter":{"bool":{"minimum_should_match" : 1,"should":['
    for i, (k, v) in enumerate(f.items()):
        if i > 0:
            s += ','
        s += '{"terms":{"' + k + '":{"index":"nbs","id":"' + v + '","path":"nbs"}}}'
    s += ']}}}},"functions": ['
    for i, (k, v) in enumerate(r.items()):
        if i > 0:
            s += ','
        s += '{"script_score":{"script":{"id":"hd64","params":{"field":"' + str(k) + '","subcode":' + str(
            v) + '}}},"weight":1}'
    s += '],"boost_mode":"sum","score_mode":"sum"}}}'
    r = requests.post(es_url + "/" + es_index + "/_search?size=" + str(max_res), data=s,
                      headers={"Content-Type": "application/json"}, verify=False).text
    return r


if __name__ == "__main__":
    assert int2binstr(binstr2int("1111111111111110")) == "1111111111111110"
    assert uint2binstr(binstr2uint("1111111111111110")) == "1111111111111110"

    assert reorder_code("00011110", [3, 2, 1, 0], order="asc") == "10110100"
    assert reorder_code("00011110", [1, 2, 3, 0], order="asc") == "10000111"
    assert reorder_code("00011110", [0, 1, 2, 3], order="asc") == "00011110"
    assert reorder_code("00011110", [3, 2, 1, 0]) == "00011110"
    assert reorder_code("00011110", [1, 2, 3, 0]) == "11010010"
    assert reorder_code("00011110", [0, 1, 2, 3]) == "10110100"
    print(binstr2int("1111111111111110"))
    print(binstr2int("11111111111111111111111111111110"))
    # x = BitArray(bin="1111111111111110").int
    # # print (BitArray(bin="0000000000000001").int)
    m = gen_masks(8, 3)
    # #print (len(x))
    print(len(get_nbs(12, m)))
    #  print (",".join(get_nbs(12,m).astype(str)))
    # #print ("Masks:", x)
