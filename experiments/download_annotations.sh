cd csv
echo "Downloading Image IDs (training) ..."
wget -N https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv 
echo "Downloading Image IDs (validation) ..."
wget -N https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
echo "Downloading Image IDs (test) ..."
wget -N https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv
echo "Downloading Metadata (Class Names) ..."
wget -N https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv 
echo "Downloading Metadata (Trainable Classes) ..."
wget -N https://storage.googleapis.com/openimages/v6/oidv6-classes-trainable.txt
echo "Downloading Human-verified labels (training) ..."
wget -N https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv
echo "Downloading Human-verified labels (validation) ..."
wget -N https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv
echo "Downloading Human-verified labels (test) ..."
wget -N https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv
echo "Downloading Machine-generated labels (training) ..."
wget -N https://storage.googleapis.com/openimages/v5/train-annotations-machine-imagelabels.csv
echo "Downloading Machine-generated labels (validation) ..."
wget -N https://storage.googleapis.com/openimages/v5/validation-annotations-machine-imagelabels.csv
echo "Downloading Machine-generated labels (validation) ..."
wget -N https://storage.googleapis.com/openimages/v5/test-annotations-machine-imagelabels.csv
# cd ..

#echo "Downloading and extracting Open Images v2 labels ..."
#wget https://storage.googleapis.com/openimages/2017_07/classes_2017_07.tar.gz
#tar -xzf classes_2017_07.tar.gz
#cp 2017_07/classes-trainable.txt csv/labels_v2.csv
#cp 2017_07/class-descriptions.csv csv/labeldescriptions_v2.csv
#rm classes_2017_07.tar.gz
#rm -rf 2017_07
#
#wget https://storage.googleapis.com/openimages/2017_07/annotations_human_2017_07.tar.gz
#tar -xzf annotations_human_2017_07.tar.gz
#wget https://storage.googleapis.com/openimages/2017_07/annotations_machine_2017_07.tar.gz
#tar -xzf annotations_machine_2017_07.tar.gz
rm -rf *.tar.gz
cd ..
