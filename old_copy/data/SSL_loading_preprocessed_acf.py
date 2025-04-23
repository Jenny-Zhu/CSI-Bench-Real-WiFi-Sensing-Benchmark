import torch
import os
import mat73
from torch.utils.data import Dataset

class SSLACFDatasetMAT(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []
        # root_dir = data_dir + "/OW_HumanNonhuman/"+ experiment+"/"

        # folder_name = os.path.join(data_dir, 'ssl_l/')
        file_list = os.listdir(data_dir)
        file_list = [i for i in file_list if i.endswith('5_half.mat')]

        for file_path in file_list:
            samples = mat73.loadmat(os.path.join(data_dir, file_path))['X']
            samples_tensor = torch.from_numpy(samples).float()
            if samples_tensor.dim() == 2:
                samples_tensor = samples_tensor.unsqueeze(0)
            self.samples.append(samples_tensor)

        # self.samples = torch.stack(self.samples, dim=0)

        self.samples = torch.unsqueeze(torch.cat(self.samples, dim=0), dim=-3)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index]