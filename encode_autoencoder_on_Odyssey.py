from __future__ import print_function
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from sklearn.metrics import average_precision_score
import pickle
import itertools
import math
import glob
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from encode_autoencoder_app_IO import *

# Configuring device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

datainput_path = "/n/scratchlfs/xiaoleliu_lab/cchen/Cistrome_imputation/encode/data/DNase_train/"
model_path = "/n/scratchlfs/xiaoleliu_lab/cchen/Cistrome_imputation/encode/model/"

def get_file_names():
    os.chdir(datainput_path)
    total_files = glob.glob("*joint_data_25bp_version.hdf5")
    total_files = [i[:i.index("_chr")] for i in total_files]
    file_names = list(set(total_files))
    file_names = [i for i in file_names if total_files.count(i)>=15]
    return(file_names)

def prepare_optparser():
    usage = "usage: %prog -i train_file_name -t test_file_name"
    description = "Extracting DNase features using auto-encoder"
    optparser = OptionParser(version="%prog v1.00", description=description, usage=usage, add_help_option=False)
    optparser.add_option("-h","--help",action="help",help="Show this help message and exit.")
    optparser.add_option("-p","--file_path",dest="file_path",type="string",
                         help="file_path")
    optparser.add_option("-m","--model_path",dest="model_path",type="string",
                         help="model_path")
    (options,args) = optparser.parse_args()
    return(options)

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.conv1d = nn.Conv1d(2, 32, 21, stride=1, padding=10)
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.MaxPool1d(3, stride=3),
            nn.Conv1d(32, 8, 3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.MaxPool1d(3, stride=3)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 32, 3, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 2, 3, stride=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1d(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def trainNet(model,num_epochs, batch_size,learning_rate,model_name,file_names,train_chromosomes,test_chromosomes):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),eps=1e-08, lr=learning_rate)

    # Loss record
    loss_record = open("/%s/Loss_record_%s.txt" %(model_path,model_name),'wt')
    test_record = open("/%s/Test_record_%s.txt" %(model_path,model_name),'wt')

    latest_loss = 0

    for epoch in range(1, num_epochs+1):
        # Train the model
        model.train()
        train_loss = 0.0
        test_loss = 0.0

        for file_name in file_names:
            for train_chromosome in train_chromosomes:
                train_loader = get_train_loader_hdf5(datainput_path,file_name,train_chromosome,batch_size)
                train_loader_iter = iter(train_loader) #https://stackoverflow.com/questions/53280967/pytorch-nextitertraining-loader-extremely-slow-simple-data-cant-num-worke
                for i in range(len(train_loader)):
                    try:
                        images = next(train_loader_iter)
                    except StopIteration:
                        train_loader_iter = iter(train_loader)
                        images = next(train_loader_iter)

                    images = images.to(device) # Dimentions of images: batch_size, channels, length

                    # Forward pass
                    outputs = model(images)
                    loss = torch.sqrt(criterion(outputs, images))

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()

                    #clipping_value = 0.00001#arbitrary number of your choosing
                    #torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)

                    optimizer.step()
                    train_loss += loss.item()*images.size(0)

                    del loss,outputs,images
                    torch.cuda.empty_cache()
        loss_record.write("%s\n" %str(train_loss))

        # validation part
        model.eval()
        with torch.no_grad():
            for file_name in file_names:
                for test_chromosome in test_chromosomes:
                    test_loader = get_test_loader_hdf5(datainput_path,file_name,test_chromosome,batch_size)
                    test_loader_iter = iter(test_loader) #https://stackoverflow.com/questions/53280967/pytorch-nextitertraining-loader-extremely-slow-simple-data-cant-num-worke
                    for i in range(len(test_loader)):
                        try:
                            images = next(train_loader_iter)
                        except StopIteration:
                            train_loader_iter = iter(train_loader)
                            images = next(train_loader_iter)
                        images = images.to(device) # Dimentions of images: batch_size, channels, length

                        # Forward pass
                        outputs = model(images)
                        loss = torch.sqrt(criterion(outputs, images))
                        test_loss += loss.item()*images.size(0)

                        del loss,outputs,images
                        torch.cuda.empty_cache()

            test_record.write("%s\n" %str(test_loss))

        if epoch>3 and test_loss > (latest_loss*1.01):
            break
        else:
            latest_loss = test_loss

def main():
    opts=prepare_optparser()

    # Settings:
    datainput_path = opts.file_path
    model_path = opts.model_path
    model_name = opts.model_name

    # Settings:
    train_chromosomes = ["chr2","chr3","chr4","chr5","chr6","chr7","chr9","chr10","chr11","chr12","chr13"]
    test_chromosomes = ["chr14","chr15","chr16"]

    num_epochs = 20
    batch_size = 512
    learning_rate = 0.000005

    model = ConvAutoencoder()

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model.to(device)

    parameters = {'batch_size':batch_size,'learning_rate': learning_rate,'num_epochs': num_epochs,'model_name': model_name}

    file_names = get_file_names()
    trainNet(model, file_names=file_names,train_chromosomes = train_chromosomes,test_chromosomes = test_chromosomes,**parameters)

    torch.save(model.state_dict(), '/%s/%s_model.ckpt' %(model_path,model_name))
    torch.save(model, '/%s/%s_model.tmp' %(model_path,model_name)) # Save the whole model

    model.load_state_dict(torch.load('/%s/%s_model.ckpt' %(model_path,model_name), map_location=lambda storage, loc: storage),strict=False)
    filter_weight = (model.conv1d.weight.data.numpy())
    pickle.dump(filter_weight,file=open("/%s/filter_weight_%s.p" %(model_path,model_name),'wb'))

if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        sys.stderr.write("User interrupt me! ;-) Bye!\n")
        sys.exit(0)
