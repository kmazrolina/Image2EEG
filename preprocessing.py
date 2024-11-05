'''
This script prepares data for model training.

1) It calculates mean and std for all subjects individually.

    During model training, EEG signals for each subject
    will be transformed in the following way:

        eeg = eeg-mean[sub_index])/std[sub_index]

    This effectively standardizes the EEG signal segment, 
    making it centered around zero with a standard deviation of one

2) It creates list of imagefiles
    preprocessed_eeg_training[0,:,:,:] will correspond to the image /00001_aardvark/aardvark_01b.jpg, 
    preprocessed_eeg_training[1,:,:,:] will correspond to the image /00001_aardvark/aardvark_02s.jpg 
    etc...
'''

import numpy as np
import os

eeg_dir = 'preprocessed_eeg_data'
get_data_dir = 'GetData'
img_train_dir = 'Images/training_images'
img_test_dir = 'Images/test_images'

def eeg_prep():
    # EEG mean and std for each subject
    preprocessed_mean_overall = np.array(10) # 10 subjects
    preprocessed_std_overall = np.array(10)

    for i in range(1):
        subject_dir = f'sub-{str(i+1).zfill(2)}'
        data_path = os.path.join(eeg_dir, subject_dir, 'preprocessed_eeg_training.npy')
        
        eeg_data_train = np.load('preprocessed_eeg_data/sub-01/preprocessed_eeg_training.npy', allow_pickle=True)
        
        mean = np.mean(eeg_data_train.tolist()['preprocessed_eeg_data'])
        std = np.std(eeg_data_train.tolist()['preprocessed_eeg_data'])
        
        preprocessed_mean_overall = np.append(preprocessed_mean_overall, mean)
        preprocessed_std_overall= np.append(preprocessed_std_overall, std)
        print(f'\tEEG ata from subject {i+1} processed')

    print('Saving...')
    np.save(os.path.join(get_data_dir,'preprocessed_mean_overall.npy'), preprocessed_mean_overall)
    np.save(os.path.join(get_data_dir,'preprocessed_std_overall.npy'), preprocessed_std_overall)

def img_prep():
    #Image list

    training_imgpaths = np.array(1654*10) # classes x num of images per class
    test_imgpaths = np.array(200) 
    
    for dirpath, dirnames, filenames in os.walk(img_train_dir):
        for filename in filenames:
            # Construct the full path to the file
            full_path = os.path.join(dirpath, filename)
            training_imgpaths = np.append(training_imgpaths, full_path)

    print('\t Train images processed')

    for dirpath, dirnames, filenames in os.walk(img_test_dir):
        for filename in filenames:
            # Construct the full path to the file
            full_path = os.path.join(dirpath, filename)
            test_imgpaths = np.append(test_imgpaths, full_path)
    print('\t Test images processed')
    print('Saving...')
    np.save(os.path.join(get_data_dir,'test_imgpaths.npy'), test_imgpaths)
    np.save(os.path.join(get_data_dir,'training_imgpaths.npy'), training_imgpaths)

            


if __name__ == '__main__':
    print("EEG PREP")
    eeg_prep()
    print("SUCCESS")
    

    print("IMAGE PREP")
    img_prep()
    print("SUCCESS")
   
    

