'''
This script prepares data for model training.

1) It calculates mean and std for all subjects individually.

    During model training, EEG signals for each subject
    will be transformed in the following way:

        eeg = eeg-mean[sub_index])/std[sub_index]

    This effectively standardizes the EEG signal segment, 
    making it centered around zero with a standard deviation of one

2) Gets only 20 eeg time points from stimulus onset to 200ms after stimulus
    And averages data of an image across 4 repetitions of the same image
   (!) Overrites the original dataset with 100 time points per image
    original data shape (16540 x 4 x 17 x 100)
    preprocessed data shape (16540 x 17 x 20)
   
3) It creates list of imagefiles
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
    # EEG mean and std for each subject, for every eeg channel 
    preprocessed_mean_overall = [] #2d array (10,17) subjects x eeg chanels
    preprocessed_std_overall = []

    for i in range(10):
        print(f'\tProcessing EEG data from subject {i+1}...')
        subject_dir = f'sub-{str(i+1).zfill(2)}'
        train_data_path = os.path.join(eeg_dir, subject_dir, 'preprocessed_eeg_training.npy')
        test_data_path = os.path.join(eeg_dir, subject_dir, 'preprocessed_eeg_test.npy')

        eeg_data_train = np.load(train_data_path, allow_pickle=True).tolist()['preprocessed_eeg_data']
        eeg_data_test = np.load(test_data_path, allow_pickle=True).tolist()['preprocessed_eeg_data']

        ## Resampling eeg
        print(eeg_data_test.shape)
        # Take only 200ms after stimulus onset
        #  time steps are every 10ms, and there is 20 steps before stimulus onset
        eeg_data_train = eeg_data_train[:,:,:,20:40] 
        eeg_data_test = eeg_data_test[:,:,:,20:40]
        print(eeg_data_test.shape)

        # Average acorss image repetitions
        eeg_data_train = np.mean(eeg_data_train, 1)
        eeg_data_test = np.mean(eeg_data_test, 1)
        print(eeg_data_test.shape)

        print('Saving...')
        #Overrites original data
        np.save(train_data_path, eeg_data_train)
        np.save(test_data_path, eeg_data_test)


        ## Std and Mean across images
        mean_train = np.mean(eeg_data_train, 0)
        mean_test = np.mean(eeg_data_test, 0)
        mean_all = (mean_train + mean_test)/ 2  

        std_train = np.std(eeg_data_train, 0)
        std_test = np.std(eeg_data_test, 0)
        std_all = (std_train + std_test) / 2
        
        preprocessed_mean_overall.append(mean_all)
        preprocessed_std_overall.append(std_all)


        print(f'\tEEG data from subject {i+1} processed.')

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
    

    # print("IMAGE PREP")
    # img_prep()
    # print("SUCCESS")
   
    

