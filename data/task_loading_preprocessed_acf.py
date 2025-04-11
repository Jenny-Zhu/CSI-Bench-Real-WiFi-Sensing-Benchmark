from torch.utils.data import Dataset, DataLoader,random_split
import os
import torch
import mat73

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
  assert task in ['HumanNonhuman', 'FourClass', 'NTUHumanID', 'NTUHAR', 'Widar','HumanMotion','ThreeClass','DetectionandClassification','Detection']


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
    # four_class = ['Human', 'Pet', 'IRobot','Fan']
    # for ind, val in enumerate(four_class):
    #   if val in file_name:
    #     label = ind+1
    #     break

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


  elif task == 'Detection':
    label_dict = {'NoMotion': 0, 'Motion': 1}

    if 'nomotion' in file_name:
        label=label_dict['NoMotion']
        print('NoMotion labeled')
    elif 'Human' or 'Fan' or'Pet' or'IRobot' in file_name:
        label=label_dict['Motion']
        print('Motion labeled')
        print(label)
    else:
      print('Unrecognize class type for  ' +file_name )


  elif task == 'DetectionandClassification':

    label_dict = {'NoMotion': 0, 'HumanMotion': 1,
                  'PetMotion':2, 'IRobotMotion':3,
                  'FanMotion':4,}

    if 'nomotion' in file_name:
        label=label_dict['NoMotion']
        print('NoMotion labeled')
    elif 'Human' in file_name:
        label=label_dict['HumanMotion']
        print('Human Motion labeled')
    elif 'Pet' in file_name:
        label=label_dict['PetMotion']
        print('Pet Motion labeled')
    elif 'IRobot' in file_name:
        label=label_dict['IRobotMotion']
        print('IRobot Motion labeled')
    elif 'Fan' in file_name:
        label=label_dict['FanMotion']
        print('Fan Motion labeled')
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

  elif task == 'NTUHumanID':
    tester_list = ['001', '002', '003', '004', '005',
               '006', '007', '008', '009', '010',
               '011', '012', '013', '014', '015']
    label = None  # Initialize label
    for ind, val in enumerate(tester_list):
        if val in file_name:
            label = ind
            break
    if label is None:
        print('Unrecognized class type for ' + file_name)

  elif task == 'NTUHAR':
    tester_list = ['run', 'walk', 'box', 'circle', 'clean', 'fall']
    label = None  # Initialize label
    for ind, val in enumerate(tester_list):
        if val in file_name:
            label = ind
            break
    if label is None:
        print('Unrecognized class type for ' + file_name)

  elif task == 'Widar':
    tester_list = ['PP', 'Sw', 'Cl', 'Sl', 'DNH', 'DOH','DRH','DTH',
                   'DZH','DZ','DN','DO','Dr1','Dr2','Dr3','Dr4','Dr5',
                   'Dr6','Dr7','Dr8','Dr9','Dr10']
    label = None  # Initialize label
    for ind, val in enumerate(tester_list):
        if val in file_name:
            label = ind
            break
    if label is None:
        print('Unrecognized class type for ' + file_name)
  else:
        pass
  return label

class ACFDatasetOW_HM3_MAT(Dataset):
    def __init__(self, data_dir, task):
        self.samples = []
        self.labels = []
        folder_name = data_dir

        file_list = os.listdir(folder_name)
        file_list = [i for i in file_list if i.endswith('5_half.mat')]
        # file_list = [x for x in file_list if "Human" in x]

        if task == 'ThreeClass':
          file_list  = [x for x in file_list if "Fan" not in x]
        for file_path in file_list:
            samples = mat73.loadmat(os.path.join(folder_name, file_path))['X']
            samples_tensor = torch.from_numpy(samples).float()
            self.samples.append(samples_tensor)
            label = generate_labels(task, file_path)
            for i in range(samples_tensor.shape[0]):
              self.labels.append(label)

        self.samples = torch.unsqueeze(torch.cat(self.samples, dim=0),dim=-3)


    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

class DatasetNTU_MAT(Dataset):
    def __init__(self, data_dir, task):
        self.samples = []
        self.labels = []
        folder_name = data_dir

        file_list = os.listdir(folder_name)
        if task =='Widar':
            file_list = [i for i in file_list if i.endswith('.mat')]
        else:
            file_list = [i for i in file_list if i.endswith('amp.mat')]
        # file_list = [x for x in file_list if "Human" in x]

        # if task == 'NTUHumanID':
        #   file_list  = [x for x in file_list]
        for file_path in file_list:
            samples = mat73.loadmat(os.path.join(folder_name, file_path))['X']
            samples_tensor = torch.from_numpy(samples).float()
            if samples_tensor.shape[0] == 250:
                samples_tensor = samples_tensor.unsqueeze(0)
            self.samples.append(samples_tensor)
            label = generate_labels(task, file_path)
            for i in range(samples_tensor.shape[0]):
              self.labels.append(label)

        self.samples = torch.unsqueeze(torch.cat(self.samples, dim=0),dim=-3)


    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]
