'''
Downloads data from THINGSEEG2 dataset 

Images
EEG - train and test sets for 10 subjects

Full datased size ~=50GB 
'''
import os
import zipfile
import urllib.request

def setup_dir(dir_path):
    if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")

def setup_dirs(weights_dir, get_data_dir, preprocessed_eeg_dir, images_dir):
   
    setup_dir(get_data_dir) 
    setup_dir(preprocessed_eeg_dir)
    setup_dir(images_dir)

    for i in range(1, 11): 
        dir_path = os.path.join(weights_dir, f'sub-{str(i+1).zfill(2)}') 
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
    
    get_data_dir = 'GetData'
    preprocessed_eeg_dir = 'preprocessed_eeg_data'
    images_dir = 'Images'

    setup_dirs( get_data_dir, preprocessed_eeg_dir, images_dir)

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


    #download and unzip the data
    for entry in metadata:
        download_data(entry['url'], os.path.join(entry['dir'], entry['filename']))
        if (entry['filename']).endswith('.zip'):
           unzip_data(os.path.join(entry['dir'], entry['filename']), entry['dir'])  

    # unzip data for all subjects
    for i in range(10):
        filename = f'sub-{str(i+1).zfill(2)}.zip'
        unzip_data(os.path.join(preprocessed_eeg_dir, filename), preprocessed_eeg_dir)  

    
    
  