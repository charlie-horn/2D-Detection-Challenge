def get_dataset_generator(C, data_directory, get_image_output_length, class_mapping, mode="train"):
    from waymo_open_dataset.utils import range_image_utils
    from waymo_open_dataset.utils import transform_utils
    from waymo_open_dataset.utils import  frame_utils
    import numpy as np
    import cv2
    from waymo_open_dataset import dataset_pb2 as open_dataset
    import helpers
    import tensorflow as tf
    import os
    from frcnn.keras_frcnn.data_generators import get_new_img_size, calc_rpn
    remote_input_files = helpers.get_remote_file_names(data_directory)
    input_file = ""
    i=0
    
    for remote_input_file in remote_input_files:
        try:
            os.remove(input_file)
        except:
            pass
        input_file = helpers.get_remote_file(remote_input_file, "./input_files")
        # Change this to copy all tfrecord files and read them all in once large instance is available
        dataset = tf.data.TFRecordDataset(input_file)

        # Skipping image shuffling and image augmentation for now
        # TODO try adding this in and comparing results
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            for image in frame.images:
                i+=1
                # Get bboxes for calc_rpn
                img_data = {'bboxes': []}

                # Create cv2 img object
                image_contents_array = np.fromstring(image.image, np.uint8)
                img = cv2.imdecode(image_contents_array, cv2.IMREAD_COLOR) 
                input_img = cv2.imdecode(image_contents_array, cv2.IMREAD_COLOR)

                for camera_labels in frame.camera_labels:
                    if camera_labels.name != image.name:
                        continue
                    for label in camera_labels.labels:
                        x1 = int(label.box.center_x - label.box.width/2)
                        x2 = int(label.box.center_x + label.box.width/2)
                        y1 = int(label.box.center_y - label.box.length/2)
                        y2 = int(label.box.center_y + label.box.length/2)
                        img_data['bboxes'].append({'class': class_mapping[label.type],
                                                  'x1': x1,
                                                  'x2': x2,
                                                  'y1': y1,
                                                  'y2': y2})
                        cv2.rectangle(input_img,(x1, y1), (x2, y2), (0, 0, 255),2)
                #cv2.imwrite('./input_images/{}.png'.format(i), input_img)
                try:
                    (rows, cols, _) = img.shape
                    img_data['width'] = cols
                    img_data['height'] = rows
                    img_min_side = float(C.im_size)

                    if cols <= rows:
                        ratio = img_min_side/cols
                        new_height = int(ratio * rows)
                        new_width = int(img_min_side)
                    else:
                        ratio = img_min_side/rows
                        new_width = int(ratio * cols)
                        new_height = int(img_min_side)

                    (resized_width, resized_height) = (new_width, new_height)
                    img_data['ratio'] = ratio
                    img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                    try:
                        y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data, cols, rows, resized_width, resized_height, get_image_output_length)
                    except:
                        continue

                    # Zero-center by mean pixel, and preprocess image

                    img = img[:,:, (2, 1, 0)]  # BGR -> RGB
                    img = img.astype(np.float32)
                    img[:, :, 0] -= C.img_channel_mean[0]
                    img[:, :, 1] -= C.img_channel_mean[1]
                    img[:, :, 2] -= C.img_channel_mean[2]
                    img /= C.img_scaling_factor

                    img = np.transpose(img, (2, 0, 1))
                    img = np.expand_dims(img, axis=0)

                    y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

                    img = np.transpose(img, (0, 2, 3, 1))
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                    yield np.copy(img), [np.copy(y_rpn_cls),np.copy(y_rpn_regr)], img_data, frame.context, image

                except Exception as e:
                    print(e)
                    continue

