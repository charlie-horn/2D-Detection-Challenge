
# Get the Waymo Open Dataset library
git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od

# Get latest frcnn library
mkdir frcnn
git clone https://github.com/charlie-horn/keras-frcnn.git frcnn

mkdir input_files

ln -s ./frcnn/keras_frcnn keras_frcnn

mkdir weights
mkdir results_imgs
mkdir input_images
mkdir output

#gcloud auth login chorn6300@gmail.com

pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-1-0==1.2.0
pip3 install gcloud
pip3 install opencv-python
pip3 install pandas
pip3 install keras

