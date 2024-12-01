import math
from collections import OrderedDict
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn.functional as F
import os
import random
from tqdm.auto import tqdm

torch.set_default_dtype(torch.float32)

# transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):
    def forward(self, x):
        return x

class CORblock_S(nn.Module):
    scale = 4

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()
        self.times = times
        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)
        self.output = Identity()

        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)
        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)
            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)
            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)
            x += skip
            x = self.nonlin3(x)
        return self.output(x)

def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),
            ('output', Identity())
        ]))),
        ('V2', CORblock_S(64, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model

class Decoder(nn.Module):
    def __init__(self, n_input, realnet):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(n_input, 512)
        self.fc_it = nn.Linear(512, 25088)
        self.fc_v4 = nn.Linear(128, 50176)
        self.fc_v2 = nn.Linear(128, 100352)
        self.fc_v1 = nn.Linear(128, 200704)
        self.activation = nn.ReLU()
        self.realnet = realnet

    def forward(self, eeg_features):
        features = self.fc(eeg_features)
        features = self.activation(features)

        it_features = self.fc_it(features)
        it_features = it_features.view(-1, 512, 7, 7)

        v4_features = self.fc_v4(features)
        v4_features = v4_features.view(-1, 256, 14, 14)

        v2_features = self.fc_v2(features)
        v2_features = v2_features.view(-1, 128, 28, 28)

        v1_features = self.fc_v1(features)
        v1_features = v1_features.view(-1, 64, 56, 56)

        v1_output = self.realnet.module.V1(v1_features)
        v2_output = self.realnet.module.V2(v2_features)
        v4_output = self.realnet.module.V4(v4_features)
        it_output = self.realnet.module.IT(v4_output)
        output_image = self.realnet.module.decoder(it_output)

        return output_image

class Data4Model(Dataset):
    def __init__(self, state='training', sub_index=1, transform=None):
        imgs = np.load(f'GetData/{state}_imgpaths.npy').tolist()
        eeg = np.load(f'preprocessed_eeg_data/sub-{str(sub_index).zfill(2)}/preprocessed_eeg_{state}.npy', allow_pickle=True)
        self.imgs = imgs
        self.eeg = torch.tensor(eeg).float()
        self.transform = transform

    def __len__(self):
        return min(len(self.imgs), len(self.eeg))

    def __getitem__(self, item):
        img = self.transform(Image.open(self.imgs[item]).convert('RGB'))
        eeg = self.eeg[item]
        return eeg, img

def train_decoder(decoder, train_loader, test_loader, optimizer, criterion, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = decoder.to(device)
    best_loss = float("inf")

    for epoch in range(num_epochs):
        decoder.train()
        train_loss = 0.0
        for eeg, imgs in tqdm(train_loader):
            eeg, imgs = eeg.to(device), imgs.to(device)
            optimizer.zero_grad()
            outputs = decoder(eeg)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        decoder.eval()
        test_loss = 0.0
        with torch.no_grad():
            for eeg, imgs in tqdm(test_loader):
                eeg, imgs = eeg.to(device), imgs.to(device)
                outputs = decoder(eeg)
                loss = criterion(outputs, imgs)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(decoder.state_dict(), "best_decoder_model.pth")
            print("Saved Best Model")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cornet = CORnet_S().to(device)

    decoder = Decoder(n_input=128, realnet=cornet)
    train_data = Data4Model(state='training', sub_index=1, transform=transform)
    test_data = Data4Model(state='testing', sub_index=1, transform=transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

    train_decoder(decoder, train_loader, test_loader, optimizer, criterion, num_epochs=25)
