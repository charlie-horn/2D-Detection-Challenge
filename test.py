def test():
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
    #print(device_lib.list_local_devices())

    # Helpers
    import generator
    import helpers
    C = config.Config()
    C.rpn_weights_path = "weights/resnet.hdf5"
    C.class_weights_path = "weights/resnet.hdf5"

    output_file = "output/predictions.bin"

    class_mapping = {"zero_class": 0,
                     "TYPE_VEHICLE": 1,
                     "TYPE_PEDESTRIAN": 2,
                     "three_class": 3,
                     "TYPE_CYCLIST": 4,
                     "bg": 5}

    class_mapping = {v: k for k, v in class_mapping.items()}

    remote_folder = "gs://waymo_open_dataset_v_1_2_0_individual_files/validation/"

    dataset_generator = generator.get_dataset_generator(C, remote_folder, nn.get_img_output_length, class_mapping, mode="validate")

    img_input = Input(shape=(None,None,3))
    roi_input = Input(shape=(None,4))
    feature_map_input = Input(shape=(None, None, 1024))

    shared_layers = nn.nn_base(img_input, trainable=True)

    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    model_rpn.load_weights(C.rpn_weights_path, by_name=True)
    model_classifier.load_weights(C.class_weights_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    idx = 0

    bbox_threshold = 0.6

    while True:
      X, Y, img_data, context, image = next(dataset_generator)

      # get the feature maps and output from the RPN
      [Y1, Y2, F] = model_rpn.predict(X)
      R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

      # convert from (x1,y1,x2,y2) to (x,y,w,h)
      R[:, 2] -= R[:, 0]
      R[:, 3] -= R[:, 1]

      # apply the spatial pyramid pooling to the proposed regions
      bboxes = {}
      probs = {}
      for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)

        if ROIs.shape[1] == 0:
          break

        if jk == R.shape[0]//C.num_rois:
          #pad R
          curr_shape = ROIs.shape
          target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
          ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
          ROIs_padded[:, :curr_shape[1], :] = ROIs
          ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
          ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

        for ii in range(P_cls.shape[1]):
          if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
            continue

          cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

          if cls_name not in bboxes:
            bboxes[cls_name] = []
            probs[cls_name] = []

          (x, y, w, h) = ROIs[0, ii, :]

          cls_num = np.argmax(P_cls[0, ii, :])
          try:
            (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
            tx /= C.classifier_regr_std[0]
            ty /= C.classifier_regr_std[1]
            tw /= C.classifier_regr_std[2]
            th /= C.classifier_regr_std[3]
            x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
          except:
            pass
          bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
          probs[cls_name].append(np.max(P_cls[0, ii, :]))

      helpers.create_prediction(output_file, context, image, bboxes, probs, img_data['ratio'])

      all_dets = []
      img=X[0]
      img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
      #print("Output Shape:",img.shape)
      drawn = False
      for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
          drawn = True
          (x1, y1, x2, y2) = new_boxes[jk,:]

          #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255),2)

      idx += 1
      if drawn:
        print("Drawn")
        img = cv2.resize(img, (int(img.shape[1]//img_data['ratio']), int(img.shape[0]//img_data['ratio'])), interpolation=cv2.INTER_CUBIC)
        img = img[:,:, (2, 1, 0)]
        img[:, :, 0] += C.img_channel_mean[0]
        img[:, :, 1] += C.img_channel_mean[1]
        img[:, :, 2] += C.img_channel_mean[2]

        #cv2.imwrite('./results_imgs/{}.png'.format(idx), img)
