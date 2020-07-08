# Script performing the inference and evaluation of smt-models
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import os.path

from multi_config import load_configurations
import Display.localizationPlot as lp
from new_createDataset import *
from SoftNetworkModel import SoftNetwork
from Math.NadarayaWatson import *
import mir_eval
import pretty_midi


version = "soft_oneSided_" + str(int(sys.argv[2])) + "ms"
print("<<<<<", version, ">>>>>")

_, test_files = newSplitDataset()

f1_list = []
pre_list = []
rec_list = []

if sys.argv[1]!="False":
    file_range = range(0,60)
else:
    file_range = range(45,46)

# GPU
import os
if len(sys.argv)>3:
    if int(sys.argv[3])==0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"


for idx_file in file_range:
    print(idx_file,"--------")
    infer_files = test_files[idx_file:idx_file + 1]
    print("----", infer_files[0], "----")

    if not os.path.isfile('infer/loose' + version + "_" + infer_files[0].split("/")[-1].replace('.wav', '.npy')):

        prediction_list = np.zeros([0, 2])

        # Load configuration
        config = load_configurations(version)
        assert(config['extension']==version)

        config['augmentation_factor'] = 7
        config['cores'] = 40


        threshold = 0.35
        print("---", config['extension'], "---")

        # Load out-of-sample data
        print('>> Load Dataset...')
        x_out_tmp, _, y_out_raw, stretch_factor_out, file_list_out = generateSplitDataset(infer_files, config,
                                                                                          infer=True)
        x_out = np.zeros([x_out_tmp.shape[0], x_out_tmp.shape[1], 2 * x_out_tmp.shape[2]])
        x_out[:, :, :x_out_tmp.shape[2]] = x_out_tmp
        x_out[:, 1:, x_out_tmp.shape[2]:] = np.diff(x_out_tmp, n=1, axis=1)

        pad = config['start_channel']
        y_out_raw = y_out_raw[:, pad:pad + config['n_channel'], :]

        # Obtain Predictions
        with tf.Session() as sess:
            # Restore model
            softModel = SoftNetwork(config)
            softModel.restore(sess)

            # Single extract score
            pp = softModel.predictHM(sess, x_out)
            _, _ = lp.localizationPlot(pp, y_out_raw, n_samples=20, dist_threshold=config['tolerence'], factor=1,
                                       bias=config['temporal_bias'], decimals=7)
            plt.close()

            sess.close()
        softModel.reset()


        # np.save('TitleImage/HH', pp)
        # print('#######################################')

        # Ensembling score
        print('Ensembling')
        pp_trans = np.transpose(pp.reshape(
            [pp.shape[0] // config['augmentation_factor'], config['augmentation_factor'], pp.shape[1], pp.shape[2]]),
            [1, 0, 2, 3])
        stretch_factor_trans = np.transpose(stretch_factor_out.reshape(pp.shape[0] // config['augmentation_factor'], config['augmentation_factor']),[1,0])

        pp_ensemble = np.zeros([pp_trans.shape[1], pp_trans.shape[2], pp_trans.shape[3]])
        for sequence_idx in range(pp_trans.shape[1]):
            for channel_idx in range(pp_trans.shape[2]):
                summ = 0
                for ensemble_idx in range(pp_trans.shape[0]):
                    line = np.interp(np.arange(pp_trans.shape[3]),
                                       np.arange(pp_trans.shape[3]) * stretch_factor_trans[ensemble_idx,sequence_idx],
                                       pp_trans[ensemble_idx, sequence_idx, channel_idx, :])
                    summ+=line
                pp_ensemble[sequence_idx,channel_idx,:] = summ/pp_trans.shape[0]


        idx_display = np.unravel_index(pp_trans[:,:,:,:].argmax(), pp_trans[:,:,:,:].shape)
        plt.figure()
        plt.subplot(3,1,1)
        for kk in range(pp_trans.shape[0]):
            plt.plot(pp_trans[kk,idx_display[1],idx_display[2],:])
        plt.subplot(3, 1, 2)
        for kk in range(pp_trans.shape[0]):
            line = np.interp(np.arange(pp_trans.shape[3]),
                               np.arange(pp_trans.shape[3]) * stretch_factor_trans[kk,0],
                               pp_trans[kk,idx_display[1],idx_display[2], :])
            plt.plot(line)
        plt.subplot(3, 1, 3)
        plt.plot(pp_ensemble[idx_display[1],idx_display[2]])
        plt.savefig('HM_ensembling')
        plt.close()



        # if sys.argv[1]=="False":
        #     print("Normal ensembling")
        #     pp_ensemble = softModel.ensembling(pp_trans, stretch_factor_out, ensembling_factor, suppression_field)
        # else:
        #     print("Parallel ensembling")
        #     pp_ensemble = softModel.ensemblingParallel(pp_trans, stretch_factor_out, ensembling_factor, suppression_field)
        #
        # plt.figure()
        # _, _ = lp.localizationPlot(pp_ensemble, y_out_raw[::config['augmentation_factor'], :, :], n_samples=10,
        #                            dist_threshold=config['tolerence'],
        #                            factor=1, bias=config['temporal_bias'], decimals=7)
        # plt.close()

        # np.save('TitleImage/KD_data', pp_ensemble)

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



        # Gaussian filter
        import scipy
        signal = np.zeros([201])  # 21
        signal[len(signal) // 2] = 1
        filt = scipy.ndimage.filters.gaussian_filter(signal, config['smoothing_width'])

        # Peak-picking
        predictions = pp_final[0,:,:] / np.max(filt)

        final_prediction = np.zeros([predictions.shape[0], predictions.shape[1]])
        for idx in range(predictions.shape[0]):
            while np.max(
                    predictions[idx, :] - scipy.ndimage.filters.gaussian_filter(final_prediction[idx, :], config['smoothing_width']) / np.max(filt)) >= threshold:
                series = predictions[idx, :] - scipy.ndimage.filters.gaussian_filter(
                    final_prediction[idx, :], config['smoothing_width']) / np.max(filt)
                xx_max = np.argmax(series)
                n_points = max(int(np.round(series[xx_max] * 1.2)),1)
                range_min = xx_max
                range_max = xx_max

                for pp in range(n_points - 1):
                    if series[range_min - 1] > series[range_max + 1]:
                        range_min -= 1
                    else:
                        range_max += 1

                final_prediction[idx, range_min:range_max + 1] = 1

        plt.figure()
        plt.plot(pp_final[0,idx_display[2],:])
        plt.plot(y_final[0,idx_display[2],:]*np.max(filt))
        plt.plot(final_prediction[idx_display[2],:]*np.max(filt), 'r')
        plt.xlim([500,1000])
        plt.savefig('HM_ensembling2')
        plt.close()

        pp_final_cleaning = final_prediction[np.newaxis,:,:]



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

        # Check alignment
        plt.figure()
        plt.plot(yy[0, 0, :] - y_final[0, 0, :])
        plt.close('all')



        # From original data
        plt.figure()
        fig, _ = lp.localizationPlot(pp_final_cleaning[:, :, :], yy[:, :, :], n_samples=pp_final_cleaning.shape[0],
                                     dist_threshold=config['tolerence'],
                                     factor=1, bias=0, decimals=7)
        plt.close()

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
            # plt.savefig('plt/inference/' + config['extension'])
        except:
            pass
        plt.close('all')

        print("---", config['extension'], "---")

        tmp_list = pp_list[0]  # supports only one file per run...
        tmp_list[:, 1] += config['start_channel']
        prediction_list = tmp_list # np.concatenate([prediction_list, tmp_list], axis=0)

        ################################ End of indent
        ## Final performance
        label_tmp = np.loadtxt(infer_files[0].replace('.wav', '.txt'), skiprows=1)
        label_raw = label_tmp[:, [0, 2]]

        fig, _ = lp.localizationPlotList([prediction_list], [label_raw], decimals=7, bias=-0.025, n_samples=1)
        plt.savefig('infer/images/' + version + "_" + infer_files[0].split("/")[-1].replace('.wav', '.png'))
        plt.close('all')

        np.save('infer/loose' + version + "_" + infer_files[0].split("/")[-1].replace('.wav', '.npy'), prediction_list)


        # Final MIR Score
        pre, rec, f1, _ = (
            mir_eval.transcription.precision_recall_f1_overlap(
                np.concatenate([label_raw[:, 0:1], label_raw[:, 0:1]+1], axis=1),
                pretty_midi.note_number_to_hz(label_raw[:,1]),
                np.concatenate([prediction_list[:, 0:1], prediction_list[:, 0:1]+1], axis=1),
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
