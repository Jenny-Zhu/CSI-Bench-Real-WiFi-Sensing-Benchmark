import scipy.io
import numpy as np
import json

def load_files_by_type(task,label,set_type):
    with open('dataset/metadata/'+task+'_dataset_metadata_2.json', 'r') as f:
        metadata = json.load(f)
    return [entry['file_path'] for entry in metadata if entry['label']== label and entry['set_type'] == set_type]


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