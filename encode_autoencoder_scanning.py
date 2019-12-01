from __future__ import print_function
import torch
import torch.utils.data as data
import h5py
from pathlib import Path
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

from encode_autoencoder_on_Odyssey import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = "/n/scratchlfs/xiaoleliu_lab/cchen/Cistrome_imputation/encode/model/"
scan_path = "/n/scratchlfs/xiaoleliu_lab/cchen/Cistrome_imputation/encode/data/DNase_scanning/raw_data/"
target_path = "/n/scratchlfs/xiaoleliu_lab/cchen/Cistrome_imputation/encode/data/DNase_scanning/scan_result/"
chromosomes = ["chr%s" %str(i) for i in (list(range(1,23))+["X","Y"])]

def get_file_names():
    os.chdir(scan_path)
    total_files = glob.glob("*sorted_hg19_25bpbin_bwaverage_transformed_*.hdf5")
    total_files = [i[:i.index(".corrected_sorted")] for i in total_files]
    file_names = list(set(total_files))
    return(file_names)

def filter_input(model_name):
    model = ConvAutoencoder()
    model.load_state_dict(torch.load('/%s/%s_model.ckpt' %(model_path,model_name), map_location=lambda storage, loc: storage),strict=False)
    return(model.conv1d.weight.data)

def scanning_supp(x,filters):
    #conv1d = nn.Conv1d(2, 16, 9, stride=1, padding=0)
    conv1d = nn.Conv1d(2, 32, 21, stride=1, padding=0)
    conv1d.weight.data = filters
    maxpool = nn.MaxPool1d(kernel_size = 8, stride = 2)
    return(maxpool(conv1d(x))[0])

def coordinates_creation(size):
    coordinates = []
    for i in range(size):
        coordinates.append([i*50,i*50+200])
    return(coordinates)

def DNase_scanning(firster_layer_filters,file_names):
    for file_name in file_names:
        for chromosome in chromosomes:
            f = h5py.File("/%s/%s.corrected_sorted_hg19_25bpbin_bwaverage_transformed_%s.hdf5" %(scan_path,file_name,chromosome),'r')
            DNase_signal = f["normalized_signal"][()]
            DNase_signal = np.nan_to_num(DNase_signal)
            #padding_signal = np.zeros((2,4))
            padding_signal = np.zeros((2,10))
            original_DNase_size = DNase_signal.shape[1]

            #----
            DNase_signal = np.concatenate((padding_signal,DNase_signal),axis=1)
            DNase_signal = np.concatenate((DNase_signal,padding_signal),axis=1)
            DNase_signal_input = DNase_signal.reshape(1,DNase_signal.shape[0],DNase_signal.shape[1])
            DNase_signal_input = torch.from_numpy(DNase_signal_input).float()
            #----

            with h5py.File("%s/%s.corrected_sorted_hg19_25bpbin_bwaverage_transformed_%s_scanned_with_autoencoder_v4.hdf5" % (target_path,file_name,chromosome),"w") as outfile:
                scanning_signal = scanning_supp(DNase_signal_input,firster_layer_filters)
                outfile.create_dataset("DNase_feature_scanning", dtype=np.float32,data = np.transpose(scanning_signal.detach().numpy()))

                coordinates = coordinates_creation(scanning_signal.shape[1])
                outfile.create_dataset("coordinates", dtype=np.float32,data = np.array(coordinates))

def main():
    file_names = get_file_names()
    model_name = "encode_autoencoder_v4"
    firster_layer_filters = filter_input(model_name)
    DNase_scanning(firster_layer_filters,file_names)

main()
