import csv
import os
import argparse
from util import allowed_file

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Create a list of images in a directory and subdirectories.')

    parser.add_argument(
        '--images_dir',
        default="images",
        type=str,
        help='Directory containing images'
    )

    parser.add_argument(
        '--allowed_files',
        choices=["png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"], nargs="*",
        default=["png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"],
        help='Image extensions'
    )

    parser.add_argument(
        '--filter_prefix',
        default="",
        type=str,
        help='Consider only files beginning with prefix'
    )

    parser.add_argument(
        '--output',
        default="images.csv",
        type=str,
        help='Output file'
    )

    args = parser.parse_args()

    image_dir = args.images_dir
    allowed_files = args.allowed_files
    output = args.output

    pathes = []

    print("""Reading images...""")

    with open(output, "w") as f:
        csv_writer = csv.writer(f)
        for root, dirs, files in os.walk(image_dir):
            for filename in files:
                if (allowed_file(filename, allowed_ext=allowed_files)):
                    if filename.startswith(args.filter_prefix):
                        path = os.path.join(os.path.relpath(root, image_dir), filename)
                        csv_writer.writerow(path)
