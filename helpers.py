def get_remote_file_names(data_directory):
  list_command = "gsutil ls " + data_directory
  file_names_string_output = subprocess.check_output(list_command, shell=True).decode("utf-8")
  file_names_list = file_names_string_output.split("\n")[:-1]
  return file_names_list

def get_remote_file(remote_file, target_folder):
  cp_command = "gsutil cp " + remote_file + " " + target_folder
  file_to_copy = target_folder + "/" + os.path.basename(remote_file)
  if not os.path.exists(file_to_copy):
    print("Copying", file_to_copy)
    subprocess.call(cp_command, shell=True)
  return file_to_copy
