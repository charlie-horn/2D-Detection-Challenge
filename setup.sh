git config --global user.email "charlie.horn@queensu.ca"
git config --global user.name "charlie-horn"

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

sudo apt-get install python3-pip
pip3 install --upgrade pip
#pip3 install waymo-open-dataset-tf-2-1-0==1.2.0
pip3 install waymo-open-dataset-tf-2-1-0
pip3 install gcloud
pip3 install opencv-python
pip3 install pandas
pip3 install keras

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1604-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb

export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.1/bin:$PATH
