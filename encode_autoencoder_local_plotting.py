import numpy as np
from collections import defaultdict
import seaborn as sns
import pickle
import pandas as pd
import matplotlib.pyplot as plt

data_path = "/Users/chen-haochen/Dropbox/Cistrome_imputation/DNase_feature/data/"
figure_path = "/Users/chen-haochen/Dropbox/Cistrome_imputation/DNase_feature/figures/"

def DNase_filter_plotting():
    filters = pickle.load(open("/%s/filter_weight_encode_autoencoder_v4.p" %(data_path),'rb'))
    '''
    fig, axes = plt.subplots(8,2, figsize=(20,20))

    subplot_num = 0

    for i in range(8):
        for j in range(2):
            ax = axes[i, j]
            filter = filters[subplot_num,:,:]
            df = pd.DataFrame(filter,index=["forward","reverse"])
            sns.heatmap(df,xticklabels=False, yticklabels=False,vmin=-0.5,vmax=0.5,cmap='coolwarm')
            subplot_num += 1
    '''

    fig, axn = plt.subplots(8, 4, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([.85, .3, .03, .4])

    for i, ax in enumerate(axn.flat):
        sns.heatmap(filters[i,:,:], ax=ax,
                    cbar=i == 0,
                    vmin=-0.5,vmax=0.5,cmap='coolwarm',
                    cbar_ax = None if i else cbar_ax)

    fig.tight_layout(rect=[0, 0, .8, 1])

    plt.savefig("/%s/autoencoder_filter_autoencoder_v4.png" %(figure_path))
    plt.close()

def main():
    DNase_filter_plotting()

main()
