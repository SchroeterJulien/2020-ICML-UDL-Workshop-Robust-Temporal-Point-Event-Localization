# Script performing the inference and evaluation

import matplotlib.pyplot as plt
import mir_eval
import numpy as np
import os
import os.path
import pretty_midi
import sys
import tensorflow as tf

from config import load_configurations
import Display.localizationPlot as lp
from createDataset import *
from SoftNetworkModel import SoftNetwork
from Math.NadarayaWatson import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Version name
version = "soft_model_" + str(sys.argv[2]) + "ms"
print("<<<<<", version, ">>>>>")

# Load test set
_, test_files = newSplitDataset()

# Placeholder for scores
f1_list = []
pre_list = []
rec_list = []

if sys.argv[1]!="False":
    file_range = range(0,60)
else:
    file_range = range(45,46) # The extract 45 is computed without multiprocess for memory usage reasons

for idx_file in file_range:
    infer_files = test_files[idx_file:idx_file + 1]
    print("----", infer_files[0], "----")

    if not os.path.isfile('infer/' + version + "_" + infer_files[0].split("/")[-1].replace('.wav', '.npy')):
        # If not already computed, computed predictions

        # Placeholder for predictions
        prediction_list = np.zeros([0, 2])

        # Load configuration
        config = load_configurations(version)
        assert(config['extension']==version)

        # Overwrite augmentation factor and number of cores
        config['augmentation_factor'] = 7
        config['cores'] = 40
        ensembling_factor = 0.02 * config['augmentation_factor']

        # Other settings (kept fixed, no hyperparameter optimization was performed)
        suppression_field = 20
        threshold = 0.35

        with tf.Session() as sess:
            # Restore model
            softModel = SoftNetwork(config)
            softModel.restore(sess)

            # Load out-of-sample data
            print('>> Load Dataset...')
            x_out_tmp, _, y_out_raw, stretch_factor_out, file_list_out = generateSplitDataset(infer_files, config,
                                                                                          infer=True)
            # Add derviative of the mel-spectrogram
            x_out = np.zeros([x_out_tmp.shape[0], x_out_tmp.shape[1], 2 * x_out_tmp.shape[2]])
            x_out[:, :, :x_out_tmp.shape[2]] = x_out_tmp
            x_out[:, 1:, x_out_tmp.shape[2]:] = np.diff(x_out_tmp, n=1, axis=1)

            # Select channel range
            pad = config['start_channel']
            y_out_raw = y_out_raw[:, pad:pad + config['n_channel'], :]

            # Single extract score
            pp = softModel.predict(sess, x_out)
            _, _ = lp.localizationPlot(pp, y_out_raw, n_samples=20, dist_threshold=config['tolerence'], factor=1,
                                       bias=config['temporal_bias'], decimals=7)
            plt.close()

            sess.close()
        softModel.reset()


        # Extract ensembling (i.e. put together all extracts from data augmentation)
        print('Ensembling')
        pp_trans = np.transpose(pp.reshape(
            [pp.shape[0] // config['augmentation_factor'], config['augmentation_factor'], pp.shape[1], pp.shape[2]]),
            [1, 0, 2, 3])
        
        if sys.argv[1]=="False":
            print("Normal ensembling") # Just for extract 45
            pp_ensemble = softModel.ensembling(pp_trans, stretch_factor_out, ensembling_factor, suppression_field)
        else:
            print("Parallel ensembling")
            pp_ensemble = softModel.ensemblingParallel(pp_trans, stretch_factor_out, ensembling_factor, suppression_field)

        plt.figure()
        _, _ = lp.localizationPlot(pp_ensemble, y_out_raw[::config['augmentation_factor'], :, :], n_samples=10,
                                   dist_threshold=config['tolerence'],
                                   factor=1, bias=config['temporal_bias'], decimals=7)
        plt.close()


        # Reassemble overlapping samples
        _start_extract = 0

        y_ensemble = y_out_raw[::config['augmentation_factor'], :, :]
        file_list_out_ensemble = file_list_out[::config['augmentation_factor']]
        y_pasted = np.zeros([len(np.unique(file_list_out_ensemble)), pp_ensemble.shape[1], 200000], np.float32)
        pp_pasted = np.zeros([len(np.unique(file_list_out_ensemble)), pp_ensemble.shape[1], 200000], np.float32)
        ww = np.zeros([len(np.unique(file_list_out_ensemble)), pp_ensemble.shape[1], 200000], np.float32)
        file_out_unique = []
        previous_source = ""
        idx_source = -1
        for ii in range(len(file_list_out_ensemble)):
            if file_list_out_ensemble[ii] == previous_source:
                idx_start += 100  # 0.5 step
            else:
                idx_start = 0
                idx_source += 1
                previous_source = file_list_out_ensemble[ii]
                file_out_unique.append(previous_source)

            y_pasted[idx_source, :, idx_start:idx_start + y_ensemble[ii, :, _start_extract:].shape[1]] += y_ensemble[ii,
                                                                                                          :,
                                                                                                          _start_extract:]
            pp_pasted[idx_source, :, idx_start:idx_start + pp_ensemble[ii, :, _start_extract:].shape[1]] += pp_ensemble[
                                                                                                            ii,
                                                                                                            :,
                                                                                                            _start_extract:]
            ww[idx_source, :,
            idx_start:idx_start + pp_ensemble[ii, :, _start_extract:300 + _start_extract].shape[1]] += 1

        # Normalize
        pp_final = pp_pasted[:, :, np.sum(ww, axis=(0, 1)) > 0] / ww[:, :, np.sum(ww, axis=(0, 1)) > 0]
        y_final = y_pasted[:, :, np.sum(ww, axis=(0, 1)) > 0] > 0

        # Load labels from file
        yy = np.zeros([pp_final.shape[0], pp_final.shape[1], pp_final.shape[2]])
        yy_list = []
        for jj in range(yy.shape[0]):
            label_tmp = np.loadtxt(file_out_unique[jj].replace('.wav', '.txt'), skiprows=1)
            label_raw = label_tmp[:, [0, 2]]
            label_raw = label_raw[[x >= pad and x < pad + config['n_channel'] for x in label_raw[:, 1]], :]
            label_raw[:, 1] -= pad
            for kk in range(label_raw.shape[0]):
                yy[jj, int(label_raw[kk, 1]), int(label_raw[kk, 0] * 200)] += 1

            yy_list.append(label_raw)


        # Final prediction cleaning
        pp_final = pp_pasted[:, :, np.sum(ww, axis=(0, 1)) > 0] / ww[:, :, np.sum(ww, axis=(0, 1)) > 0]
        pp_final_cleaning = np.zeros([pp_final.shape[0], pp_final.shape[1], pp_final.shape[2]])
        for ii in range(pp_final_cleaning.shape[0]):
            for jj in range(pp_final_cleaning.shape[1]):
                for tt in range(pp_final_cleaning.shape[2]):
                    if pp_final[ii, jj, tt] > 0:
                        if np.sum(pp_final[ii, jj, tt:tt + suppression_field]) >= threshold:
                            pp_final_cleaning[ii, jj, tt] = 1
                            pp_final[ii, jj, tt:tt + suppression_field] = 0


        # From original data
        plt.figure()
        fig, _ = lp.localizationPlot(pp_final_cleaning[:, :, :], yy[:, :, :], n_samples=pp_final_cleaning.shape[0],
                                     dist_threshold=config['tolerence'],
                                     factor=1, bias=config['temporal_bias'], decimals=7)
        plt.close()

        # Detection array to list of detections
        pp_list = []
        for ii in range(pp_final.shape[0]):
            triggers = np.zeros([0, 2])
            for jj in range(pp_final.shape[1]):
                list_hits = np.where(pp_final_cleaning[ii, jj])[0] / 200
                triggers = np.concatenate([triggers, np.concatenate(
                    [list_hits[:, np.newaxis], np.array([jj] * len(list_hits))[:, np.newaxis]], axis=1)])
            pp_list.append(triggers)

        try:
            fig, _ = lp.localizationPlotList(pp_list, yy_list, decimals=7, bias=-0.025, n_samples=1)
        except:
            pass
        plt.close('all')

        print("---", config['extension'], "---")

        tmp_list = pp_list[0]  # supports only one file per run...
        tmp_list[:, 1] += config['start_channel']
        prediction_list = tmp_list

        # Final performance (using our function)
        label_tmp = np.loadtxt(infer_files[0].replace('.wav', '.txt'), skiprows=1)
        label_raw = label_tmp[:, [0, 2]]

        fig, _ = lp.localizationPlotList([prediction_list], [label_raw], decimals=7, bias=-0.025, n_samples=1)
        plt.savefig('infer/images/' + version + "_" + infer_files[0].split("/")[-1].replace('.wav', '.png'))
        plt.close('all')

        np.save('infer/' + version + "_" + infer_files[0].split("/")[-1].replace('.wav', '.npy'), prediction_list)

        # Final MIR Score
        prediction_list[:,0] += 0.025
        pre, rec, f1, _ = (
            mir_eval.transcription.precision_recall_f1_overlap(
                np.concatenate([label_raw[:, 0:1], label_raw[:, 0:1]+1], axis=1),
                pretty_midi.note_number_to_hz(label_raw[:,1]),
                np.concatenate([prediction_list[:, 0:1]-0.030, prediction_list[:, 0:1]+1], axis=1),
                pretty_midi.note_number_to_hz(prediction_list[:,1]),
                offset_ratio=None))

        f1_list.append(f1)
        pre_list.append(pre)
        rec_list.append(rec)

        print(f1, pre, rec)
        print(np.mean(f1_list), np.mean(pre_list), np.mean(rec_list), idx_file)
        print('---------------------------')
    else:
        print("<<< Already exists!")
