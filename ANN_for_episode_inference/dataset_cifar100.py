import torch
from torch.utils.data import Dataset
import numpy as np


class Dataset(Dataset):
    def __init__(self, data_path, class_list):
        self.inputs = np.load(data_path+'/inputs.npy')
        self.labels = np.load(data_path+'/labels.npy')

        self.class_list = class_list

        self.inputs_list = []
        self.labels_list = []

        for i in range(len(class_list)):
            index = np.where(self.labels == class_list[i])[0][:500]
            self.inputs_list.append(self.inputs[index])
            self.labels_list.append(self.labels[index])

        self.inputs = np.concatenate(self.inputs_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0)

        self.inputs = torch.from_numpy(self.inputs).float()
        self.labels = torch.from_numpy(self.labels).long()

        print('inputs: ', self.inputs.shape)
        print('labels: ', self.labels.shape)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]
