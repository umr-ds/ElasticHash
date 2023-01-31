import xml.etree.ElementTree as ET
import urllib.request
import os

download_dir = "../res/download/"
num_images = 1000
struct_path = os.path.join(download_dir, "struct.xml")
status_path = os.path.join(download_dir, "status.xml")
struct_url = "http://www.image-net.org/api/xml/structure_released.xml"
status_url = "http://www.image-net.org/api/xml/ReleaseStatus.xml"


def download():
    response = urllib.request.urlopen(struct_url)
    data = response.read()
    with open(struct_path, "wb") as f:
        f.write(data)
    response = urllib.request.urlopen(status_url)
    data = response.read()
    with open(status_path, "wb") as f:
        f.write(data)


def get_counts():
    d = {}
    root = ET.parse(status_path).getroot()
    for s in root.findall(".//synset"):
        d[s.attrib["wnid"]] = int(s.attrib["numImages"])
    return d


def get_leaf_nodes(counts):
    leaves = []
    root = ET.parse(struct_path).getroot()

    def get_leaves(leaves, node):
        if len(list(node)) > 0:
            for n in node:
                get_leaves(leaves, n)
        else:
            if node.tag == "synset":
                wnid = node.attrib["wnid"]
                if (wnid in counts) and (counts[wnid] > num_images):
                    leaves += [wnid]
        return leaves

    return get_leaves(leaves, root)


if __name__ == "__main__":
    # download()
    counts = get_counts()
    leaves = get_leaf_nodes(counts)
    with open("../res/leaves.txt", "w") as f:
        f.write("\n".join(leaves))
