import os
import glob
import json
from sklearn.model_selection import train_test_split

def generate_metadata(base_path,locations,task,metadatafile_path):
    metadata = []
    file_dict = {}

    # Collect files from all locations
    for location in locations:
        pattern = os.path.join(base_path,location,"Data", "*", "*", "*", "*csi*.mat")
        file_paths = glob.glob(pattern, recursive=True)

        # Organize files by subject
        for path in file_paths:
            parts = path.split(os.sep)
            subject = parts[7]  # Assuming 'Subject' is in the 4th place of the path
            if subject not in file_dict:
                file_dict[subject] = []
            file_dict[subject].append(path)

    # Split files and prepare metadata
    for subject, files in file_dict.items():
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        metadata += [{"file_path": file, "task": task, "label": subject, "set_type": "train"} for file in train_files]
        metadata += [{"file_path": file, "task": task, "label": subject, "set_type": "test"} for file in test_files]

    # Save metadata to a JSON file
    with open(metadatafile_path, 'w') as f:
        json.dump(metadata, f, indent=4)

# Example locations
task = "OW_HM3"
base_path = "E:\Dataset\Dataset_OW\DatasetHP\HM3_HP"
locations = [
                # "HM3.0 2_8_2023 HealthPod",
                # "HM3.0 2_8_2023 Test House HealthPod Merged File",
                # "HM3.0 4_3_2023 old office Health Pod",
                "HM3.0 4_13_2023 Test House Health Pod Merged",
                "HM3.0 6_15_2023 Test House Health Pod merged",
                "HM3.0 6_27_2023 Test House Health Pod merged",
                ]
metadatafile_path = 'C:\Guozhen\Code\Github\WiFiSSL\dataset\metadata\\'+task+'_dataset_metadata_2.json'
generate_metadata(base_path,locations,task,metadatafile_path)


##Use#####################
# def load_files_by_type(task,label,set_type):
#     with open(metadatafile_path, 'r') as f:
#         metadata = json.load(f)
#     return [entry['file_path'] for entry in metadata if entry['label']== label and entry['set_type'] == set_type]
#
# task = 'OW_HM3'
# label = 'Fan'
# train_files = load_files_by_type(task,label,'train')
# test_files = load_files_by_type(task,label,'test')