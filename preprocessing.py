'''
This script prepares data for model training.
It calculates mean and std for all subjects individually.

During model training, EEG signals for each subject
will be transformed in the following way:

    eeg = eeg-mean[sub_index])/std[sub_index]

This effectively standardizes the EEG signal segment, 
making it centered around zero with a standard deviation of one
'''

import numpy as np
import os

if __name__ == '__main__':

    preprocessed_mean_overall = np.array(10) # 10 subjects
    preprocessed_std_overall = np.array(10)

    eeg_dir = 'preprocessed_eeg_data'
    get_data_dir = 'GetData'

    for i in range(1):
        subject_dir = f'sub-{str(i+1).zfill(2)}'
        data_path = os.path.join(eeg_dir, subject_dir, 'preprocessed_eeg_training.npy')
        
        eeg_data_train = np.load('preprocessed_eeg_data/sub-01/preprocessed_eeg_training.npy', allow_pickle=True)
        
        mean = np.mean(eeg_data_train.tolist()['preprocessed_eeg_data'])
        std = np.std(eeg_data_train.tolist()['preprocessed_eeg_data'])
        
        preprocessed_mean_overall = np.append(preprocessed_mean_overall, mean)
        preprocessed_std_overall= np.append(preprocessed_std_overall, std)

    np.save(os.path.join(get_data_dir,'preprocessed_mean_overall.npy'), preprocessed_mean_overall)
    np.save(os.path.join(get_data_dir,'preprocessed_std_overall.npy'), preprocessed_std_overall)


    

