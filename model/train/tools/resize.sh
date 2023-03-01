echo "" >processed.txt
find ImageNet/unpacked_more_than_1000/ -name '*.JPEG' | while read f; do
  convert $f -resize 300x300 $f
  echo $f >>processed.txt
done
find Places/data_large_extra/ Places/val_large Places/data_large -name '*.jpg' | while read f; do
  convert $f -resize 300x300 $f
  echo $f >>processed.txt
done

# find ImageNet/unpacked_more_than_1000/ -name '*.JPEG' -execdir mogrify -resize 300x {} \; ;find Places/data_large_extra/ Places/val_large Places/data_large -name '*.jpg' -execdir mogrify -resize 300x {} \;
#
#find . -name '*.JPEG' -execdir mogrify -resize 300x {} \;
