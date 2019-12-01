import os
import random
import glob
import subprocess as sp
import pandas as pd
import random
import numpy as np
from collections import defaultdict
import h5py


processed_TF_path = "/project/dev/cchen/data6/Cistrome_imputation/encode/TF/processed_data/"
processed_DNase_path = "/project/dev/cchen/data6/Cistrome_imputation/encode/DNase_hdf5/"
target_path = "/project/dev/cchen/data6/Cistrome_imputation/encode/data_prep/"
chromosomes = ["chr%s" %str(i) for i in (list(range(1,23))+["X","Y"])]

def return_DNase_files():
    os.chdir(processed_DNase_path )
    file_names_prefixes = glob.glob("*corrected_sorted_hg19_25bpbin_bwaverage_transformed*.hdf5")
    file_names_prefixes = list(set([i[:i.index(".corrected_sorted_hg19")] for i in file_names_prefixes]))
    return(file_names_prefixes)

def return_bins_with_TF_bindings(cell,chromosome):
    try:
        df = pd.read_csv("/%s/encode_TF_hg19_25bpbin_%s_%s_summary_with_bindings.csv" %(processed_TF_path,cell,chromosome),sep=",",header=0,index_col=0)
        return(df)
    except:
        temp = pd.DataFrame()
        return temp

def return_bins_with_TF_bindings_extended(TF_binding_bins):
    selected_bin_ranges = []
    selected_bins = []
    for bin in TF_binding_bins:
        bin_number_in_25bp = int(bin[bin.index("bin")+4:])
        for m in [-1,0,1]:
            bin_range = [(bin_number_in_25bp)-22+m,(bin_number_in_25bp)+23+m]
            if bin_range[0] >= 0:
                selected_bins.append(bin)
                selected_bin_ranges.append(bin_range)
    return(selected_bins,selected_bin_ranges)

def data_prep_for_training_25bp():
    file_names_prefixes = return_DNase_files()

    for chromosome in chromosomes:
        for file_names_prefix in file_names_prefixes:
            cell = (file_names_prefix.split("."))[1]

            TF_binding_df = return_bins_with_TF_bindings(cell,chromosome)
            TF_sample_numbers = TF_binding_df.shape[0]
            TF_binding_bins = TF_binding_df.columns.values
            selected_bins,selected_bin_ranges = return_bins_with_TF_bindings_extended(TF_binding_bins)

            if len(selected_bins) >= 500:
                DNase_f = h5py.File("/%s/%s.corrected_sorted_hg19_25bpbin_bwaverage_transformed_%s.hdf5" %(processed_DNase_path,file_names_prefix,chromosome),'r')
                normalized_signal = DNase_f["normalized_signal"][()]
                normalized_signal = np.nan_to_num(normalized_signal)

                DNase_signal_in_selected_bins = normalized_signal[:,[list(range(i[0],i[1])) for i in selected_bin_ranges]]
                DNase_signal_in_selected_bins = DNase_signal_in_selected_bins.transpose(1,0,2)

                with h5py.File("/%s/%s_%s_joint_data_25bp_version.hdf5" %(target_path,file_names_prefix,chromosome),'w') as output:
                    string_dt = h5py.special_dtype(vlen=str)
                    selected_bins = np.array(selected_bins, dtype=object)
                    output.create_dataset("center_25bp_bin", data = selected_bins, dtype=string_dt)
                    TF_samples = np.array(TF_binding_df.index.tolist(), dtype=object)
                    output.create_dataset("TF_sample_numbers", data = TF_samples, dtype=string_dt)

                    bin_ranges = output.create_dataset("included_25bp_bin_range", dtype=np.float32, shape=(len(selected_bins),2),compression='gzip', shuffle=True, fletcher32=True, compression_opts=4)
                    DNase = output.create_dataset("DNase", dtype=np.float32, shape=(len(selected_bins),2,45),compression='gzip', shuffle=True, fletcher32=True, compression_opts=4)
                    TF = output.create_dataset("TF_binding", dtype=np.float32, shape=(len(selected_bins),TF_sample_numbers),compression='gzip', shuffle=True, fletcher32=True, compression_opts=4)

                    TF[:,:] = TF_binding_df[selected_bins].T.values
                    bin_ranges[:,:] = np.array(selected_bin_ranges)
                    DNase[:,:,:] = DNase_signal_in_selected_bins

                    output.flush()

def data_examination_for_training_25bp():
    os.chdir(target_path)
    for file in glob.glob("*joint_data_25bp_version.hdf5"):
        f = h5py.File(file,'r')
        DNase = f.get('DNase')[()]
        TF = f.get('TF_binding')[()]

        print(np.isnan(DNase).any(),np.isnan(TF).any(),np.isnan(DNase).sum())

def data_prep_for_all_bins():
    file_names_prefixes = return_DNase_files()

    for chromosome in chromosomes:
        for file_names_prefix in file_names_prefixes:
            cell = (file_names_prefix.split("."))[1]
            DNase_f = h5py.File("/%s/%s.corrected_sorted_hg19_25bpbin_bwaverage_transformed_%s.hdf5" %(processed_DNase_path,file_names_prefix,chromosome),'r')
            normalized_signal = DNase_f["normalized_signal"][()]

            number_of_25bp_bin = normalized_signal.shape[1]

            center_25bp_coordinates = []
            selected_25bp_bin_ranges = []
            coordinate_ranges = []
            for i in range((number_of_25bp_bin//4)):
                if (i*4-20) >= 0 and (i*4+20)<number_of_25bp_bin:
                    selected_25bp_bin_ranges.append([i*4-20,i*4+20])
                    center_25bp_coordinates.append([i*100-100,i*100+100])
                    coordinate_ranges.append([i*100-500,i*100+500])

            #print(np.array(center_25bp_coordinates))
            DNase_signal_in_selected_bins = normalized_signal[:,[list(range(i[0],i[1])) for i in selected_25bp_bin_ranges]]
            DNase_signal_in_selected_bins = DNase_signal_in_selected_bins.transpose(1,0,2)

            with h5py.File("/%s/%s_%s_all_bins_joint_data_for_test.hdf5" %(target_path,file_names_prefix,chromosome),'w') as output:
                DNase = output.create_dataset("DNase", dtype=np.float32, shape=(len(selected_25bp_bin_ranges),2,40),compression='gzip', shuffle=True, fletcher32=True, compression_opts=4)
                surrounding_1000bp_coordinates = output.create_dataset("surrounding_1000bp_coordinates", dtype=np.float32, shape=(len(selected_25bp_bin_ranges),2),compression='gzip', shuffle=True, fletcher32=True, compression_opts=4)
                center_25bp_coordinates_entry = output.create_dataset("center_25bp_coordinates", dtype=np.float32, shape=(len(selected_25bp_bin_ranges),2),compression='gzip', shuffle=True, fletcher32=True, compression_opts=4)

                DNase[:,:,:] = DNase_signal_in_selected_bins
                center_25bp_coordinates_entry[:,:] = np.array(center_25bp_coordinates)
                surrounding_1000bp_coordinates[:,:] = np.array(coordinate_ranges)

                output.flush()


def main():
    #data_prep_for_training_25bp()
    #data_examination_for_training_25bp()
    #data_prep_for_all_bins_25bp()

main()
