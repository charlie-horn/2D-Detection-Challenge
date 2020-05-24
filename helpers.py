def get_remote_file_names(data_directory):
    import subprocess
    list_command = "gsutil ls " + data_directory
    file_names_string_output = subprocess.check_output(list_command, shell=True).decode("utf-8")
    file_names_list = file_names_string_output.split("\n")[:-1]
    return file_names_list

def get_remote_file(remote_file, target_folder):
    import subprocess
    import os
    cp_command = "gsutil cp " + remote_file + " " + target_folder
    file_to_copy = target_folder + "/" + os.path.basename(remote_file)
    if not os.path.exists(file_to_copy):
        print("Copying", file_to_copy)
        subprocess.call(cp_command, shell=True)
    return file_to_copy

def create_prediction(output_file, context, image, bboxes, probs):
    print("Context Name:", context.name)
    print("Timestamp:", image.pose_timestamp)
    print("Camera name:", image.name)
    print("Bbox:", bboxes)
    print("Probs:", probs)
    return
    objects = metrics_pb2.Objects()
    file = open(output_file, "rb")
    objects.ParseFromString(file.read())
    f.close()

    for i, bbox in enumerate(bboxes):
        o = metrics_pb2.Object()
        # The following 3 fields are used to uniquely identify a frame a prediction
        # is predicted at. Make sure you set them to values exactly the same as what
        # we provided in the raw data. Otherwise your prediction is considered as a
        # false negative.
        o.context_name = context.name

        # The frame timestamp for the prediction. See Frame::timestamp_micros in
        # dataset.proto.
        o.frame_timestamp_micros = image.pose_timestamp
        # This is only needed for 2D detection or tracking tasks.
        # Set it to the camera name the prediction is for.
        o.camera_name = image.name

        # Populating box and score.
        box = label_pb2.Label.Box()
        box.center_x = 0
        box.center_y = 0
        box.center_z = 0
        box.length = 0
        box.width = 0
        box.height = 0
        box.heading = 0
        o.object.box.CopyFrom(box)
        # This must be within [0.0, 1.0]. It is better to filter those boxes with
        # small scores to speed up metrics computation.
        o.score = 0.5

        # Use correct type.
        o.object.type = probs[i]

        objects.objects.append(o)

    # Add more objects. Note that a reasonable detector should limit its maximum
    # number of boxes predicted per frame. A reasonable value is around 400. A
    # huge number of boxes can slow down metrics computation.

    # Write objects to a file.
    file = open(output_file, 'wb')
    file.write(objects.SerializeToString())
    file.close()
    return
