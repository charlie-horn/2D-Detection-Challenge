# General modules
import os
import tensorflow as tf
import math
import numpy as np
import itertools
import subprocess
import importlib
import time
import pathlib
import pandas as pd

# Waymo
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import cv2

# Google
#from google.colab.patches import cv2_imshow
#from gcloud import storage

# Keras
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K

# FRCNN
from frcnn.keras_frcnn import resnet as nn
from frcnn.keras_frcnn import config
import frcnn.keras_frcnn.roi_helpers as roi_helpers
from frcnn.keras_frcnn import losses as losses
from frcnn.keras_frcnn.data_generators import get_new_img_size, calc_rpn

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#K.tensorflow_backend._get_available_gpus()

# Option Parser
from optparse import OptionParser

# Train and Test
import train
import test

# Helpers
from generator import *
from helpers import *

parser = OptionParser()

parser.add_option("-m", "--mode", dest="mode", help="Mode (Train or Test)")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=32)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=2000)
parser.add_option("--epoch_length", type="int", dest="epoch_length", help="Length of epochs.", default=1000)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default="./weights/model_frcnn.hdf5")
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

if not options.mode:
    parser.Error("Specify mode")

if options.mode == "train":
    train.train()

elif options.mode == "test":
    test.test()

else:
    parser.Error("Invalid mode")
