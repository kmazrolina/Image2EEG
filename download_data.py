'''
Downloads data from THINGSEEG2 dataset 

Images
EEG - train and test sets for 10 subjects

Full datased size ~=50GB 

USAGE:
python download_data.py --subject-indices 1 2 3 
'''
import os
import zipfile
import urllib.request
import argparse

get_data_dir = 'GetData'
preprocessed_eeg_dir = 'preprocessed_eeg_data'
images_dir = 'Images'
weights_dir = 'weights/ReAlnet_EGG'

# Define files to download from OSF
metadata = [
    {  
        'dir' : images_dir,
        'filename' : 'test_images.zip',
        'url': 'https://osf.io/download/znu7b/' 
    },
    {  
        'dir' : images_dir,
        'filename' : 'training_images.zip',
        'url': 'https://osf.io/download/3v527/' 
    },
    {  
        'dir' : get_data_dir,
        'filename' : 'image_metadata.npy',
        'url': 'https://osf.io/download/qkgtf/'
    },
    {  
        'dir' : preprocessed_eeg_dir,
        'filename' : 'osfstorage-archive.zip',
        'url': 'https://files.de-1.osf.io/v1/resources/anp5v/providers/osfstorage/?zip='
    }

]




def setup_dir(dir_path):
    if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

def setup_dirs(weights_dir, get_data_dir, preprocessed_eeg_dir, images_dir, sub_ids=[i for i in range(1, 11)]):
   
    setup_dir(get_data_dir) 
    setup_dir(preprocessed_eeg_dir)
    setup_dir(images_dir)

    for sub_id in sub_ids: 
        dir_path = os.path.join(weights_dir, f'sub-{str(sub_id).zfill(2)}') 
        setup_dir(dir_path)

def download_data(url, filename):
    # Download the file
    try:
        urllib.request.urlretrieve(url, filename)  # Change filename as needed
        print(f'Download of {filename} completed successfully.')
    except Exception as e:
        print(f'Failed to download {filename}. Error: {e}')


def unzip_data(zip_file_path, zip_directory='.'):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(zip_directory)  # Extract to the same directory
        print(f'Extracted: {zip_file_path} into {zip_directory}')

    # Delete the zip file
    os.remove(zip_file_path)
    print(f'Deleted zip file: {zip_file_path}')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_indices', nargs='+', type=int, default=[i for i in range (1,11)])
    parser.add_argument('--test_imgs_only', action="store_true")
    parser.add_argument('--no_eeg', action="store_true", default=False)
    args = parser.parse_args()
    sub_ids = args.subject_indices
    test_imgs_only = args.test_imgs_only
    no_eeg = args.no_eeg

    if no_eeg:
        metadata.pop(3) #remove eeg for all subjects (zip) metadata
    if test_imgs_only:
        metadata.pop(1) #remove training images metadata

    setup_dirs(weights_dir, get_data_dir, preprocessed_eeg_dir, images_dir, sub_ids)

    #download and unzip the data
    for entry in metadata:
        download_data(entry['url'], os.path.join(entry['dir'], entry['filename']))
        if (entry['filename']).endswith('.zip'):
           unzip_data(os.path.join(entry['dir'], entry['filename']), entry['dir'])  

    # unzip data for all subjects
    if not no_eeg:
        for sub_id in range(1,11):
            filename = f'sub-{str(sub_id).zfill(2)}.zip'
            unzip_data(os.path.join(preprocessed_eeg_dir, filename), preprocessed_eeg_dir)  
    


    
    
  