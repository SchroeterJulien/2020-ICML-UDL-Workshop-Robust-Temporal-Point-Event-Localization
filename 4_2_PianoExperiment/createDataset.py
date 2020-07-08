# File the functions for the dataset creation

import librosa
from multiprocessing import Pool
import numpy as np
import os
import os.path
from sklearn.externals.joblib.parallel import parallel_backend
import soundfile as sf

from AudioProcessing.mel_spectrogram import *
from google_create_dataset import * # Keep same dataset creation protocol as Hawthrone et al.

# General settings
settings = {}
settings['sample_rate'] = 44100

# Spectrogram setting # todo: pass this through config
spectrum_settings = {}
spectrum_settings['frame_size'] = 0.025
spectrum_settings['frame_stride'] = 0.005


def newSplitDataset():
    # Split dataset into test and train (as in Hawthrone)
    test_ids, test_samples = generate_test_set()
    train_samples = generate_train_set(test_ids)

    return train_samples, test_samples


def processSplitSample(input):
    # Process a full music extract

    # Process input
    file = input[0]
    infer = input[1]
    config = input[2]
    error = input[3]
    n_channel = 120

    print(file)

    # Settings
    split_length = config['split_length']
    if infer:
        split_step = 0.5
    else:
        split_step = config['split_step']

    augmentation_factor = config['augmentation_factor']
    if len(input) == 5:
        augmentation_factor = 1

    pad = config['signal_pad']
    noise = config['signal_noise']

    # Load audio extract
    signal, sr = sf.read(file)

    # Transform it to mono channel
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Check the sample rate
    assert (sr == settings['sample_rate'])

    # Additional settings # todo: pass this through config
    spectrum_settings = {}
    spectrum_settings['frame_size'] = 0.025
    spectrum_settings['frame_stride'] = 0.005  # 0.01
    spectrum_settings['number_filter'] = config['n_filter'] // 2
    spectrum_settings['NFFT'] = 4096 * 2

    # Load annotations
    label_raw = np.loadtxt(file.replace('.wav', '.txt'), skiprows=1)

    if error:
        # Load error files (apply label misalignment to the training data)
        error_file = "error_files/" + file.replace('.wav', "_" + config['extension'] + '.npy').split("/")[-1]
        error_array = np.load(error_file)
        label_raw[:, 0] += error_array

    # Placeholder for final labels and data
    final_data = np.zeros(
        [int(np.ceil(signal.shape[0] / (sr * split_step)) * augmentation_factor), config['time_steps'],
         config['n_filter'] // 2])
    labels = np.zeros([int(np.ceil(signal.shape[0] / (sr * split_step)) * augmentation_factor), config['time_steps'],
                       n_channel], np.int8)
    factor = np.zeros([int(np.ceil(signal.shape[0] / (sr * split_step)) * augmentation_factor)])

    # Loop
    idx_sample = 0
    for idx_split in range(int(np.ceil(signal.shape[0] / (sr * split_step)))):
        for idx_augmentation in range(augmentation_factor):
            if len(input) == 5:
                idx_augmentation = input[4]

            # Add some initial padding (not used in the paper)
            flag = True
            extended_signal = np.concatenate([1e-5 * np.random.randn(int(pad * sr)), signal[int(
                idx_split * sr * split_step):int(sr * (idx_split * split_step + split_length))]])

            if idx_augmentation <= 0:
                # No stretching (original data)
                stretching_factor = 1
                signal_stretch = extended_signal
            elif idx_augmentation >= 4 and not infer:
                # Stacking of two extracts as data augmentation
                stretching_factor = 1
                idx_tmp = np.random.randint(0, int(np.ceil(signal.shape[0] / (sr * split_step))) - 1)

                signal_stretch = extended_signal
                signal_new = np.concatenate([1e-5 * np.random.randn(int(pad * sr)), signal[int(
                    idx_tmp * sr * split_step):int(sr * (idx_tmp * split_step + split_length))]])

                if signal_stretch.shape[0] == signal_new.shape[0]:
                    signal_stretch += signal_new
                else:
                    flag = False

            else:
                # Temporal stretching as data augmentation
                stretching_factor = 0.75 + np.random.rand() / 4  # max(0.75 + np.random.rand() / 4, 0.9)
                signal_stretch = librosa.effects.time_stretch(extended_signal, stretching_factor)

            if flag:
                factor[idx_sample] = stretching_factor

                # Generate mel-spectrogram
                spectrum = melSpectrogram(signal_stretch + np.ones(signal_stretch.shape[0]) * noise,
                                          sr, spectrum_settings['frame_size'],
                                          spectrum_settings['frame_stride'],
                                          config['spectral_size'], spectrum_settings['NFFT'],
                                          normalized=True)

                spectrum = spectrum[:, :config['n_filter'] // 2]

                final_data[idx_sample, :spectrum.shape[0], :config['n_filter'] // 2] = spectrum
                # First order-derivative
                # Less memory requirement if done on the fly (in main script)
                ## final_data[idx_sample, 1:spectrum.shape[0], config['n_filter'] // 2:] = np.diff(spectrum, n=1,axis=0)

                # Process labels
                for ii in range(label_raw.shape[0]):
                    if label_raw[ii, 0] - split_step * idx_split >= 0 and label_raw[
                        ii, 0] - split_step * idx_split < split_length:
                        labels[idx_sample, signal2spectrumTime(
                            (np.round(label_raw[ii, 0] - split_step * idx_split, 3) + pad) / stretching_factor *
                            settings['sample_rate']), int(
                            label_raw[ii, 2])] += 1

                # For stacked extracts only
                if idx_augmentation >= 4 and not infer:
                    for ii in range(label_raw.shape[0]):
                        if label_raw[ii, 0] - split_step * idx_tmp >= 0 and label_raw[
                            ii, 0] - split_step * idx_tmp < split_length:
                            labels[idx_sample, signal2spectrumTime(
                                (np.round(label_raw[ii, 0] - split_step * idx_tmp, 3) + pad) / stretching_factor *
                                settings['sample_rate']), int(
                                label_raw[ii, 2])] += 1

                idx_sample += 1

    # Main model assumption (only used for stacked extracts and very unlikely)
    labels[labels > 1] = 1
    return final_data.astype(np.float32), labels, factor, np.array([file] * factor.shape[0])


def generateSplitDataset(file_list, config, infer=False, error=False):
    # Generate the dataset

    # Generate error files and load annotations
    if error:
        print("Create Error Files")
        for file in file_list:
            label_raw = np.loadtxt(file.replace('.wav', '.txt'), skiprows=1)

            error_file = "error_files/" + file.replace('.wav', "_" + config['extension'] + '.npy').split("/")[-1]
            error_array = config['label_noise'] * np.random.randn(label_raw.shape[0])
            np.save(error_file, error_array)

    # Placeholder for labels and data
    x_data_final = np.zeros([0, config['time_steps'], config['n_filter'] // 2], np.float32)
    y_data_final = np.zeros([0, config['time_steps'], 120], np.int8)
    factor_final = np.zeros([0])
    file_final = np.zeros([0])

    if len(file_list) > 1:
        print('multiprocessing')
        for ii in range(int(np.ceil(len(file_list) / config['cores']))):
            # Multiprocessing for improved speed
            p = Pool(config['cores'])
            if not infer:
                data_simulated = p.map(processSplitSample, [(x, infer, config, error, idx_augm) for x in
                                                            file_list[ii * config['cores']:(ii + 1) * config['cores']]
                                                            for idx_augm in np.arange(config['augmentation_factor'])])
            else:
                data_simulated = p.map(processSplitSample, [(x, infer, config, error) for x in
                                                            file_list[ii * config['cores']:(ii + 1) * config['cores']]])
            p.close()

            x_data = np.concatenate([x[0][:, :, :] for x in data_simulated], axis=0)
            y_data = np.concatenate([x[1][:, :, :] for x in data_simulated], axis=0)
            factor_list = np.concatenate([x[2] for x in data_simulated], axis=0)
            file_tmp = np.concatenate([x[3] for x in data_simulated], axis=0)

            x_data_final = np.concatenate([x_data_final, x_data], axis=0)
            y_data_final = np.concatenate([y_data_final, y_data], axis=0)
            factor_final = np.concatenate([factor_final, factor_list], axis=0)
            file_final = np.concatenate([file_final, file_tmp], axis=0)

            del data_simulated, x_data, y_data, factor_list, file_tmp
    else:
        print('single')
        x_data_final, y_data_final, factor_final, file_final = processSplitSample([file_list[0], infer, config, False])

    # Final transformations on the labels
    y_data_final = y_data_final.transpose([0, 2, 1])
    y_data_transformed = y_data_final.reshape([y_data_final.shape[0] * y_data_final.shape[1], y_data_final.shape[2]])

    Y_label = np.zeros([len(np.sum(y_data_transformed, axis=1)), config['max_occurence']])
    Y_label[np.arange(len(np.sum(y_data_transformed, axis=1))), np.sum(y_data_transformed, axis=1)] = 1
    Y_label = Y_label.reshape([y_data_final.shape[0], y_data_final.shape[1], config['max_occurence']])

    return x_data_final.astype(np.float32), Y_label.astype(np.int32), y_data_final.astype(
        np.int32), factor_final, file_final


# Utility functions
def signal2spectrumTime(time):
    """
    Converts signal time (seconds) into spectrogram bin location
    :param time: in seconds
    :return: corresponding spectrogram bin location
    """
    if time <= settings['sample_rate'] * spectrum_settings['frame_size']:
        return int(0)
    else:
        time -= settings['sample_rate'] * spectrum_settings['frame_size']
        return int(1 + time // (settings['sample_rate'] * spectrum_settings['frame_stride']))


def spectrum2signalTime(time):
    """
    Converts spectrogram bin location into signal time (seconds)
    :param time: spectrogram bin location
    :return: corresponding signal time (seconds)
    """
    if time == 0:
        return int(0)
    else:
        time -= 1
        return int(
            settings['sample_rate'] * spectrum_settings['frame_size'] + time * settings['sample_rate'] *
            spectrum_settings['frame_stride'])
