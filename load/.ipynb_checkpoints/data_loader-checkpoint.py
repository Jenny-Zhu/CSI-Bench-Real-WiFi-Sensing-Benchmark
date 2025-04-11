import os
import torch.utils.data
from data import SSLCSIDatasetSaving, SSLCSIDataset, CSIDatasetOW_HM3, SSLCSIDatasetMAT,SSLCSIDatasetHDF5,CSIDatasetOW_HM3_H5,SSLACFDatasetMAT
from data import ACFDatasetOW_HM3_MAT,DatasetNTU_MAT
from torch.utils.data import random_split,DataLoader
from util import FeatureBucketBatchSampler

# def split_support_test(support_set):
#     num_human = support_set.labels.count(1)
#     print(len(support_set.labels))
#     print(support_set.samples.shape)
#     # Split dataset into train and validation sets
#     train_size = int(0.5 * len(support_set))
#     valid_size = len(support_set) - train_size
#     support_set, test_set = random_split(support_set, [train_size, valid_size])
#     return support_set, test_set

def variable_shape_collate_fn(batch):
    # 'batch' is a list of items returned by __getitem__()
    # Each item has shape (T_i, F_i), which can be different
    # We just return the entire list as-is.
    return batch

def load_acf_data_unsupervised(data_dir,BATCH_SIZE):
    ssl_set = SSLACFDatasetMAT(data_dir)
    ssl_loader = torch.utils.data.DataLoader(ssl_set, batch_size=BATCH_SIZE, shuffle=False)

    return ssl_loader


def load_csi_data_unsupervised(data_dir,BATCH_SIZE):
    ssl_set = SSLCSIDatasetMAT(data_dir)
    sampler = FeatureBucketBatchSampler(ssl_set, batch_size=BATCH_SIZE, shuffle=True)
    ssl_loader = DataLoader(ssl_set, batch_sampler=sampler, num_workers=4)

    # ssl_loader = torch.utils.data.DataLoader(ssl_set,
    # batch_size=8,
    # collate_fn=variable_shape_collate_fn,  # custom collate
    # shuffle=True,
    # num_workers=0)

    return ssl_loader

def load_data_unsupervised(data_dir, BATCH_SIZE, win_len, sample_rate):
    ssl_set = SSLCSIDataset(data_dir, win_len, sample_rate)
    ssl_loader = torch.utils.data.DataLoader(ssl_set, batch_size=BATCH_SIZE, shuffle=False)

    return ssl_loader


def load_preprocessed_data_unsupervised(data_dir, BATCH_SIZE, win_len, sample_rate):
    # Ensure data_dir is a list, even if it's a single path
    if isinstance(data_dir, str):
        data_dir = [data_dir]  # Convert single directory string to a list
    # Initialize the dataset with the list of directories
    ssl_set = SSLCSIDatasetHDF5(data_dir, win_len, sample_rate)
    # Create a DataLoader to handle batching and shuffling
    ssl_loader = torch.utils.data.DataLoader(ssl_set, batch_size=BATCH_SIZE, shuffle=True)

    return ssl_loader


def load_data_supervised(task, BATCH_SIZE, win_len, sample_rate):
    if task == 'OW_HM3':
        data_dir = 'C:\\Guozhen\\Code\\Github\\WiFiSSL\\dataset\\metadata\\HM3_sr100_wl200'
        support_set = CSIDatasetOW_HM3_H5(data_dir,win_len, sample_rate,if_test=0)
        test_set = CSIDatasetOW_HM3_H5(data_dir,win_len, sample_rate,if_test=1)

    # elif task in ['NTU_HAR', 'NTU_HumanID']:
    #     support_set = CSIDatasetNTU(data_dir, device, win_len, sample_rate)
    #     test_set = CSIDatasetNTU(data_dir, device, win_len, sample_rate,if_test=1)
    else:
        print(f"Task unknown")
        return
    support_loader = torch.utils.data.DataLoader(support_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return support_loader, test_loader

def load_acf_supervised(data_dir,task,batch_size):
  # get the class number of different task
  classes = {'HumanNonhuman': 2, 'FourClass': 4, 'NTUHumanID': 15, 'HumanID': 4, 'HumanMotion': 3, 'ThreeClass': 3, 'DetectionandClassification':5, 'Detection':2}

  # retrive the data

  train_set = ACFDatasetOW_HM3_MAT(data_dir, task)
  num_human = train_set.labels.count(1)
  print(len(train_set.labels))
  print(train_set.samples.shape)
  # Split dataset into train and validation sets
  train_size = int(0.8 * len(train_set))
  valid_size = len(train_set) - train_size
  train_set, valid_set = random_split(train_set, [train_size, valid_size])

  # Create data loaders for train and validation sets
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False)


  return train_loader, test_loader

def load_acf_unseen_environ(data_dir,task):
  # get the class number of different task
  classes = {'HumanNonhuman': 2, 'FourClass': 4, 'NTUHumanID':15, 'HumanID': 4, 'HumanMotion': 3, 'ThreeClass': 3, 'DetectionandClassification':5, 'Detection':2}

  test_set = ACFDatasetOW_HM3_MAT(data_dir, task)
  unseen_test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
  return unseen_test_loader

def save_data_supervised(task, BATCH_SIZE, win_len, sample_rate):
    if task == 'OW_HM3':
        support_set = CSIDatasetOW_HM3(win_len, sample_rate,if_test=0)
        test_set = CSIDatasetOW_HM3(win_len, sample_rate,if_test=1)
        return support_set,test_set
    else:
        print(f"Task unknown")
        return

def load_acf_supervised_NTUHumanID(data_dir,task,batch_size):
  # get the class number of different task
  classes = {'NTUHumanID': 15, 'NTUHAR': 6, 'Widar':22,}

  # retrive the data

  train_set = DatasetNTU_MAT(data_dir, task)
  num_human = train_set.labels.count(1)
  print(len(train_set.labels))
  print(train_set.samples.shape)
  # Split dataset into train and validation sets
  train_size = int(0.8 * len(train_set))
  valid_size = len(train_set) - train_size
  train_set, valid_set = random_split(train_set, [train_size, valid_size])

  # Create data loaders for train and validation sets
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
  test_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=False)


  return train_loader, test_loader

def load_acf_supervised_NTUHumanID_fewshot(data_dir,task,batch_size):
  # get the class number of different task
  classes = {'NTUHumanID': 15, 'NTUHAR': 6, 'Widar':22,}

  # retrive the data

  train_set = DatasetNTU_MAT(data_dir, task)
  # Create data loaders for train and validation sets
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)

  return train_loader

def load_acf_unseen_environ_NTU(data_dir,task):
  # get the class number of different task
  classes = {'HumanNonhuman': 2, 'FourClass': 4, 'NTUHumanID':15, 'NTUHAR':6, 'HumanID': 4, 'Widar':22,'HumanMotion': 3, 'ThreeClass': 3, 'DetectionandClassification':5, 'Detection':2}

  test_set = DatasetNTU_MAT(data_dir, task)
  unseen_test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
  return unseen_test_loader