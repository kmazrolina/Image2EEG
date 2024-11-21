#!/usr/bin/env python
# coding: utf-8
'''
Make a single prediction with pretrained ReAlnet model.
This script saves input eeg signal and image 
and output predicted eeg into output dir.

Data for prediction is taken from test dataset (THINGSEEG2).
Choose subject 1-10 (personalized pretrained ReAlnet) to make prediction for.
'''

import torch
import torch.utils.model_zoo
import os
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from ReAlnet import CORnet_S, Encoder, Data4Model, transform


torch.set_default_dtype(torch.float32)


def build_model_from_weights(weightspath, device='cuda'):
    # this cornet will be used for getting imagenet-based outputs as the classification targets
    print("Loading cornet")
    cornet = CORnet_S().to(device)
    cornet = torch.nn.DataParallel(cornet)
    url = f'https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth'
    ckpt_data = torch.utils.model_zoo.load_url(url)
    cornet.load_state_dict(ckpt_data['state_dict'])

    # this FAnet is what we are going to test
    print("Loading cornetS")
    realnet = CORnet_S().to(device)
    realnet = torch.nn.DataParallel(realnet)
    url = f'https://s3.amazonaws.com/cornet-models/cornet_s-1d3f7974.pth'
    ckpt_data = torch.utils.model_zoo.load_url(url)
    realnet.load_state_dict(ckpt_data['state_dict'])

    print("Loading Encoder (ReAlnet)")
    encoder = Encoder(realnet, 100).to(device)
    state_dict = torch.load(weightspath, map_location=torch.device(device))
    encoder.load_state_dict(state_dict)  

    return encoder, cornet
    
def predict_batch(encoder, imgs, eeg, batch_size, pred_output_dir, device='cuda'):
    
    imgs = imgs.to(device)
    eeg = eeg.to(device)

    print("Making prediction")
    outputs, pred = encoder(imgs)
    pred = pred[:,20:40] # take only 200 ms after stimulus onset from predictions
    eeg = eeg.mean(axis=1) # mean over eeg channels

    #Image
    for i in range(batch_size):
        save_image(imgs[i], os.path.join(pred_output_dir,f"image{i}.png"))
        print("saved image ", i)

        #EEG real
        # Create time axis (each point represents 10 ms)
        time = torch.arange(0, 20) * 10  # Time in ms (0, 10, 20, ..., 190)
        # Plot the waveform
        plt.figure(figsize=(8, 4))
        plt.plot(time.numpy(), eeg[i].detach().cpu().numpy(), label='Waveform Signal')
        plt.title("EEG Signal")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        # Save the plot as an image
        plt.savefig(os.path.join(pred_output_dir,f"eeg{i}.png"), dpi=300)  # Save as a high-res PNG
        plt.close()
        print("saved eeg ", i)

        #EEG predicted
        # Create time axis (each point represents 10 ms)
        time = torch.arange(0, 20) * 10  # Time in ms (0, 10, 20, ..., 190)
        # Plot the waveform
        plt.figure(figsize=(8, 4))
        plt.plot(time.numpy(), pred[i].detach().cpu().numpy(), label='Waveform Signal')
        plt.title("Generated EEG Signal")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

        # Save the plot as an image
        plt.savefig(os.path.join(pred_output_dir,f"pred{i}.png"), dpi=300)  # Save as a high-res PNG
        plt.close()
        print("saved prediction ", i)


if __name__ == "__main__":
    print("START EVAL")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub_index = 2 # choose subject id 1-10

    # Path to model weights
    weightspath = 'weights/ReAlnet_EEG/sub-'+str(sub_index).zfill(2)+'/best_model_params.pt'
    
    # Build model with pretrained weights
    print("Loading Model Components")
    encoder, cornet = build_model_from_weights(weightspath, device)

    # Directory to store prediction output
    pred_output_dir = "predict-sub-"+str(sub_index).zfill(2) 
    os.makedirs(pred_output_dir, exist_ok=True)

    # Data for prediction set batch size as needed 
    print("Loading test data")
    batch_size = 1
    test_dataset = Data4Model(state='test', sub_index=sub_index, transform=transform)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    imgs, eeg = next(iter(test_data_loader))

    for i in range(batch_size):
        predict_batch(encoder, imgs, eeg, batch_size, pred_output_dir, device)