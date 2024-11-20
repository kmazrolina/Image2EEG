# Image to EEG
Reconstructing EEG activity from Images

## Purpose of the project
Our project about EEG Encoding is focused on advancing the understanding of human visual object recognition by utilizing EEG data in combination with deep neural networks (DNNs). This project contributes to the intersection of neuroscience and artificial intelligence, particularly in encoding and decoding the human brain’s visual responses.

## Quick Start
Pretrained weights of ReAlNet model are available at [huggingface](https://huggingface.co/kmazrolina/ReAlNet).

## Contents of the Repository:
- `download_dataset.py` - utility script to download [THINGSEEG2 dataset](https://github.com/gifale95/eeg_encoding) needed for training.
- `preprocessing.py` - preprocessing of images and eeg signals
- `ReAlnet.py` - training and testing of the model. [Source](https://github.com/jglab/ReAlnet)
- `results_visualisation.ipynb` - initial evaluation of the model. tbc. 
  
## Theory
- **EEG and Visual Processing**: Electroencephalography (EEG) is a non-invasive method to measure electrical activity in the brain. When the brain processes visual stimuli, specific patterns of activity can be detected. Decoding these patterns is crucial for understanding how humans perceive and recognize visual objects.
- **Encoding Models**: In neuroscience, encoding models predict brain activity patterns based on input stimuli. These models attempt to map the relationship between the characteristics of visual inputs (e.g., images) and their corresponding neural responses recorded by EEG.
- **Deep Neural Networks (DNNs)**: DNNs are computational architectures inspired by the brain’s structure. In this context, DNNs are used to model complex relationships between stimuli and neural activity, providing feature representations that mirror human visual processing. This allows researchers to hypothesize which DNN layers correspond to specific stages of human perception.
- **Applications**: Encoding models combined with DNNs offer insights into how visual information is processed hierarchically in the brain. They can also guide the development of brain-computer interfaces and assist in diagnosing visual or neural impairments.


## Resources
- [THINGSEEG2 dataset](https://github.com/gifale95/eeg_encoding)
- [ReAlNet](https://github.com/jglab/ReAlnet)
- [Pretrained weights of ReAlNet model](https://huggingface.co/kmazrolina/ReAlNet)
