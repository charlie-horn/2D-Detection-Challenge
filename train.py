def train(input_class_weight_path, input_rpn_weight_path, output_class_weight_path, output_rpn_weight_path, learning_rate, epoch_length, num_epochs):
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
    from frcnn.keras_frcnn import losses
    from frcnn.keras_frcnn.data_generators import get_new_img_size, calc_rpn
    #from tensorflow.python.client import device_lib

    # Help  
    import generator
    import helpers

    C = config.Config()
    C.base_rpn_weights_path = input_rpn_weight_path
    C.rpn_weights_path = output_rpn_weight_path
    C.base_class_weights_path = input_class_weight_path
    C.class_weights_path = output_class_weight_path

    class_mapping = {"zero_class": 0,
                     "TYPE_VEHICLE": 1,
                     "TYPE_PEDESTRIAN": 2,
                     "three_class": 3,
                     "TYPE_CYCLIST": 4,
                     "bg": 5}
    class_mapping_inv = {v: k for k, v in class_mapping.items()}

    remote_folder = "gs://waymo_open_dataset_v_1_2_0_individual_files/training/"

    dataset_generator = generator.get_dataset_generator(C, remote_folder, nn.get_img_output_length, class_mapping_inv)

    img_input = Input(shape=(None,None,3))
    roi_input = Input(shape=(None,4))

    shared_layers = nn.nn_base(img_input, trainable=True)

    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    try:
        print('loading RPN weights from {}'.format(C.base_rpn_weights_path))
        print('loading RPN weights from {}'.format(C.base_class_weights_path))
        model_rpn.load_weights(C.base_rpn_weights_path, by_name=True)
        model_classifier.load_weights(C.base_class_weights_path, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
                    https://github.com/fchollet/keras/tree/master/keras/applications')

    optimizer = Adam(lr=learning_rate)
    optimizer_classifier = Adam(lr=learning_rate)
    model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(class_mapping)-1)], metrics={'dense_class_{}'.format(len(class_mapping)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    iter_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf
    none_count = 0
    print('Starting training')

    for epoch_num in range(num_epochs):

        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

        while True:
            X, Y, img_data, _, _ = next(dataset_generator)
            
            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)

            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_data_format(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                none_count += 1
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue
 
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)
            
            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []
            
            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                try:
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                except Exception as e:
                    print('Exception: {}'.format(e))
                    continue
              
                sel_samples = selected_pos_samples + selected_neg_samples
              
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                print('Loss RPN regression: {}'.format(loss_rpn_regr))
                print('Loss Detector classifier: {}'.format(loss_class_cls))
                print('Loss Detector regression: {}'.format(loss_class_regr))
                print('Elapsed time: {}'.format(time.time() - start_time))
                print("None count:", none_count)
                none_count = 0
                rpn_loss = loss_rpn_cls + loss_rpn_regr
                print("Current RPN Loss:",rpn_loss)
                print("Best RPN Loss:", best_rpn_loss)
                if rpn_loss < best_rpn_loss:
                    print("# Saving RPN weights")
                    best_rpn_loss = rpn_loss
                    model_rpn.save_weights(C.rpn_weights_path)
                
                class_loss = loss_class_cls +loss_class_regr
                print("Current Classifier Loss:",class_loss)
                print("Best Clasifier Loss:",best_class_loss)
                if class_loss < best_class_loss:
                    print("# Saving Classifier weights")
                    best_class_loss = class_loss
                    model_classifier.save_weights(C.class_weights_path)
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)
                break

    print('Training complete, exiting.')
    return
