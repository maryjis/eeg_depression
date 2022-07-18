import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.signal import spectrogram
from torchvision.transforms import ToTensor
from PIL import Image


def get_device():
    return 'cuda:1' if torch.cuda.is_available() else 'cpu'

def get_np_from_edf(path_to_data, filter_params, path_to_save):
    for cls in os.listdir(path_to_data):
        for recording in os.listdir(path_to_data + f'{cls}/'):
            filename = path_to_data + f'{cls}/' + recording
            sample = mne.io.read_raw_edf(filename, verbose = False, preload = True)
            if 'EEG 2-R' in sample.ch_names:
                sample.drop_channels(['EEG 2-R'])
            sample = sample.to_data_frame(common_electrodes)[common_electrodes]
            if not os.path.isdir(path_to_save + f'{cls}'):
                os.mkdir(path_to_save + f'{cls}')
            sample.to_csv(path_to_save + f'{cls}/' + f'{recording[:-4]}.csv')

def generate_tensors_from_csv(path_to_csv, path_to_save, Fs, db = False):
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    for cls in os.listdir(path_to_csv):
        for recording_name in os.listdir(path_to_csv + f'{cls}/'):
            filename = path_to_csv + f'{cls}/' + recording_name
            data = pd.read_csv(filename)
#             data = data.iloc[:min_length]
            data_to_cat = []
            for channel in data.columns[1:]:
                signal = data[channel].to_numpy()
                freq, time, values = spectrogram(signal, Fs)
                idx = np.where((freq > 1) & (freq < 50))[0]
                freq = freq[idx]
                if db:
                    values = 20 * np.log10(values[idx])
                else:
                    values = values[idx]
                plt.figure(figsize = (3.8, 3.8), dpi = 100)
                plt.pcolormesh(time, freq, values, shading='gouraud', cmap = 'gray');
                plt.axis('off')
                plt.savefig('tensor_to_save.jpg', bbox_inches='tight', pad_inches=0, dpi = 100)
                plt.close()
                image = Image.open('tensor_to_save.jpg').resize((256, 256), resample = Image.BILINEAR)
                image = np.asarray(image)[:, :, 0]
                data_to_cat.append(ToTensor()(image))
            tensor_with_all_channels = torch.cat(data_to_cat)
            if not os.path.isdir(path_to_save + f'{cls}'):
                os.mkdir(path_to_save + f'{cls}')
            torch.save(tensor_with_all_channels, 
                       path_to_save + f'{cls}/' + f'{recording_name[:-4]}.pt')

def save_tensor(path_to_save, recording_name, cls, n, data, Fs, heigth, width, offset = False, db = True):
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    data_to_cat = []
    for channel in data.columns[1:]:
        signal = data[channel].to_numpy()
        freq, time, values = spectrogram(signal, Fs)
        idx = np.where((freq > 1) & (freq < 50))[0]
        freq = freq[idx]
        if db:
            values = 20 * np.log10(values[idx])
        else: 
            values = values[idx]
        plt.figure(figsize = (3.8, 3.8), dpi = 100)
        plt.pcolormesh(time, freq, values, shading='gouraud', cmap = 'gray');
        plt.axis('off')
        plt.savefig('tensor_to_save.jpg', bbox_inches='tight', pad_inches=0, dpi = 100)
        plt.close()
        image = Image.open('tensor_to_save.jpg').resize((heigth, width), resample = Image.BILINEAR)
        image = np.asarray(image)[:, :, 0]
        data_to_cat.append(ToTensor()(image))
    tensor_with_all_channels = torch.cat(data_to_cat)
    if not os.path.isdir(path_to_save + f'{cls}'):
        os.mkdir(path_to_save + f'{cls}')
    torch.save(tensor_with_all_channels, 
               path_to_save + f'{cls}/' + f'{recording_name[:-4]}_patch_{n}_offset={offset}.pt')
    
def generate_patched_tensors_from_csv(path_to_csv, path_to_save, Fs, window_size, offset, height, width, db):
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    for cls in os.listdir(path_to_csv):
        for recording_name in os.listdir(path_to_csv + f'{cls}/'):
            recording = pd.read_csv(path_to_csv + f'{cls}/' + recording_name)
            num_patches = np.floor((len(recording) / Fs - window_size )/(window_size - offset)).astype(int) + 1
            if offset:
                for n in range(num_patches):
                    rec1 = recording.iloc[n * window_size * Fs: (n + 1) * Fs * window_size]
                    rec2 = recording.iloc[n * Fs * window_size + Fs * offset: (n + 1) * Fs * window_size + Fs * offset]
                    if len(rec1) >= window_size:
                        save_tensor(path_to_save, recording_name, cls, n, rec1, Fs, height, width, db = db)
                    if len(rec2) >= window_size:
                        save_tensor(path_to_save, recording_name, cls, n, rec2, Fs, height, width, offset = True, db = db)
            else: 
                for n in range(num_patches):
                    rec1 = recording.iloc[n * window_size * Fs: (n + 1) * Fs * window_size]
                    if len(rec1) >= window_size:
                        save_tensor(path_to_save, recording_name, cls, n, rec1, Fs, height, width)