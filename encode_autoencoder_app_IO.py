from __future__ import print_function
import torch
import torch.utils.data as data
import h5py
from pathlib import Path
from torchvision import transforms


def get_train_loader_hdf5(datainput_path,file_name,train_chromosome,batch_size):
    train_data = H5Dataset(datainput_path,file_name,train_chromosome)
    loader_params = {'batch_size': batch_size, 'shuffle': True,'num_workers': 16, 'drop_last': True, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data,**loader_params)
    return(train_loader)

def get_test_loader_hdf5(datainput_path,file_name,test_chromosome,batch_size):
    test_data = H5Dataset(datainput_path,file_name,test_chromosome)
    loader_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 16, 'drop_last': True, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(test_data,**loader_params)
    return(test_loader)

class H5Dataset(data.Dataset):
    #https://discuss.pytorch.org/t/how-to-speed-up-the-data-loader/13740/3
    def __init__(self, datainput_path,file_name,chromosome):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File("/%s/%s_%s_joint_data_25bp_version.hdf5" %(datainput_path,file_name,chromosome),'r')
        self.DNase_signal = torch.from_numpy(h5_file.get('DNase')[()]).float()
        del h5_file

    def __getitem__(self, index):
        x = self.DNase_signal[index,:,:]
        return (x)

    def __len__(self):
        return self.DNase_signal.shape[0]
