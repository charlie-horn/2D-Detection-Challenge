cd waymo-od
bazel build waymo_open_dataset/metrics/tools/create_submission
bazel-bin/waymo_open_dataset/metrics/tools/create_submission  --input_filenames='../output/predictions.bin' --output_filename='../output/my_model/model' --submission_filename='../submission.txtpb'
tar cvf ../output/my_model.tar ../output/my_model/
gzip ../output/my_model.tar