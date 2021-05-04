import networkx as nx
import csv
import pickle as pkl
from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection as kl


def kl_partition(G, sc_len):
    def partition(G, sc_weights):
        if len(G) <= sc_len:
            sc_weights += [G.size(weight='weight')]
            return list(G.nodes)
        else:
            a_nodes, b_nodes = kl(G, partition=None, weight='weight', seed=42, max_iter=256)
            a = nx.subgraph_view(G, filter_node=lambda x: x in a_nodes)
            b = nx.subgraph_view(G, filter_node=lambda x: x in b_nodes)
            return partition(a, sc_weights) + partition(b, sc_weights)

    sc_weights = []
    p = list(map(int, partition(G, sc_weights)))
    return p, sc_weights


def compute_permutation(corr_file, bitcode_len=256, sc_len=16, out=None, method="kl"):
    # Build graph
    G = nx.Graph()
    if bitcode_len % sc_len != 0:
        raise Exception("Bitcode length (%s) not divisable by subcode length (%s)" % (
            bitcode_len, sc_len))
    with open(corr_file) as f:
        reader = csv.reader(f, delimiter="\t")
        elist = [(a, b, float(w)) for (a, b, w) in reader]
    G.add_weighted_edges_from(elist)

    p, sc_weights = kl_partition(G, sc_len)

    if out:
        with open(out, "wb") as f:
            pkl.dump([p, sc_weights], f)
    print("Permutation: ", p, " Subcode weights: ", sc_weights)
    return p, sc_weights


if __name__ == "__main__":
    print("Compute permutations...")
    p, sc_weights = compute_permutation(corr_file="./256_corr.txt", bitcode_len=256, sc_len=64)
    print(p)
    new_sc_len = 16
    p_new = p[0:new_sc_len] + p[64:64 + new_sc_len] + p[128:128 + new_sc_len] + p[192:192 + new_sc_len]
    print(p_new)
    print(len(p_new))
    with open("./64_16_from_256_perm.pkl", "wb") as f:
        pkl.dump([p_new, sc_weights], f)
    compute_permutation(corr_file="256_corr.txt", bitcode_len=256, sc_len=16, out="256_16_perm.pkl")
    # compute_permutation(corr_file="256_10000000_corr.txt", bitcode_len=256, sc_len=16, out="256_16_perm.pkl")
    # compute_permutation(corr_file="64_10000000_corr.txt", bitcode_len=64, sc_len=16, out="64_16_perm.pkl")
    # compute_permutation(corr_file="64_10000000_corr.txt", bitcode_len=64, sc_len=8, out="64_8_perm.pkl")
