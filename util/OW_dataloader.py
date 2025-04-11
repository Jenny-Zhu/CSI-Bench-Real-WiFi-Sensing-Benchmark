import torch
import os
import mat73
import numpy as np
from torch.utils.data import Dataset


def generate_labels(task,file_name):
  """ generate the class label based on the filename for different task

  Parameters
  ----------
  task: str
    The classification task. Should be one value in ['HumanNonhuman', 'FourClass', 'HumanID', 'HumanMotion']
  file_name: str
    The filename that store the data for certain class

  Returns
  ----------
  label: int
    the result encoded class label

  """
  assert isinstance(task, str)
  assert isinstance(file_name, str)
  assert task in ['HumanNonhuman', 'FourClass', 'HumanID', 'HumanMotion','ThreeClass']


  if task == 'HumanNonhuman':
    if 'Human' in file_name:
      label = 1
      print('Human labeled')
      print(label)
    else:
      label = 0
      print('Nonhuman labeled')
      print(label)

  elif task == 'FourClass':


    # Define the labels
    label_dict = {'Human': 0, 'Pet': 1, 'IRobot':2, 'Fan':3}

    if 'Human' in file_name:
        label=label_dict['Human']
        print('Human labeled')
    elif 'Pet' in file_name:
        label=label_dict['Pet']
        print('Pet labeled')
    elif 'IRobot' in file_name:
        label=label_dict['IRobot']
        print('IRobot labeled')
    elif 'Fan' in file_name:
        label=label_dict['Fan']
        print('Fan labeled')
        print(label)
    else:
      print('Unrecognize class type for  ' +file_name )

  elif task == 'ThreeClass':
    # Define the labels
    label_dict = {'Human': 0, 'Pet': 1, 'IRobot':2}

    if 'Human' in file_name:
        label=label_dict['Human']
        print('Human labeled')
    elif 'Pet' in file_name:
        label=label_dict['Pet']
        print('Pet labeled')
    elif 'IRobot' in file_name:
        label=label_dict['IRobot']
        print('IRobot labeled')
    else:
      print('Unrecognize class type for  ' +file_name )

  elif task == 'HumanID':
    tester_list = ['Andrew', 'Brain','Brendon','Dan']
    for ind,val in enumerate(tester_list):
      if val in file_name:
        label = ind+1
        break
    print('Unrecognize class type for  ' +file_name )

  elif task == 'HumanMotion':
    motion_list = ['Running','Sneaking','Walking']
    for ind,val in enumerate(motion_list):
      if val in file_name:
        label = ind+1
        break

    print('Unrecognize class type for  ' +file_name )

  else:
    pass
  return label


class OW_dataset_class(Dataset):
    def __init__(self, data_dir, task,  experiment, if_test,test_data):
        self.samples = []
        self.labels = []

        # find the corresponding data directory
        if task == 'HumanNonhuman':
            root_dir = data_dir + "/OW_HumanNonhuman/"+ experiment+"/"
        elif task == 'FourClass':
            root_dir = data_dir + "/OW_HumanNonhuman/"+ experiment+"/"
        elif task == 'ThreeClass':
            root_dir = data_dir + "/OW_HumanNonhuman/"+ experiment+"/"
        elif task == 'HumanID':
            root_dir = data_dir + '/OW_HumanID/'
        elif task == 'HumanMotion':
            root_dir = data_dir + '/OW_HAR/'

        folder_name = os.path.join(root_dir, 'train/')
        if if_test:
          if test_data == 1:
            folder_name = os.path.join(root_dir, 'test/')
          elif test_data == 2:
            folder_name = os.path.join(root_dir, 'ssl_l/')


        file_list = os.listdir(folder_name)

        if test_data ==2:
          file_list = file_list+ [i for i in file_list if i.endswith('5_half.mat')]
        else:
          file_list = [i for i in file_list if i.endswith('5_half.mat')]

        file_list  = [ x for x in file_list if "Dan" not in x ]
        # load python preprocessed .npz file
        # file_list = [i for i in file_list if i.endswith('.npz')]
        if task == 'ThreeClass':
          file_list  = [ x for x in file_list if "Fan" not in x ]
        for file_path in file_list:
            samples = mat73.loadmat(os.path.join(folder_name, file_path))['X']
            # samples = samples.transpose((0,2,1))
            # if the file list end with .npz, use np.load
            # samples = np.load(os.path.join(folder_name, file_path))['arr_0']
            samples_tensor = torch.from_numpy(samples).float()
            self.samples.append(samples_tensor)
            if test_data ==2:
              label = 0
            else:
              label = generate_labels(task, file_path)
              print("Generating dummy label 0 for SSL..")
            for i in range(samples_tensor.shape[0]):
              self.labels.append(label)



        # self.samples = torch.stack(self.samples, dim=0)
        self.samples = torch.unsqueeze(torch.cat(self.samples, dim=0),dim=-3)


    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]
