# Main script for model training

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import sys

import Display.localizationPlot as lp
from config import load_configurations
from createDataset import *
from SoftNetworkModel import SoftNetwork

if not os.path.exists('models'):
    os.mkdir('models')

if not os.path.exists('plt'):
    os.mkdir('plt')

# Load configurations
if len(sys.argv)>1:
    oneSided = sys.argv[3].lower() == 'true'
    config = load_configurations(start=sys.argv[1],noise=sys.argv[2],oneSided=oneSided)
else:
    config = load_configurations()

os.environ["CUDA_VISIBLE_DEVICES"] = config['GPU_Device']
print("---", config['extension'], "---")

# Open tensorflow session
with tf.Session() as sess:
    # Initialize model
    softModel = SoftNetwork(config)
    softModel.initialize(sess)

    print('>> Load Dataset...')
    train_files, test_files = newSplitDataset()
    print(len(train_files), len(test_files))

    print("Dataset size: ", config['dataset_size'])
    # Train set
    X_data, Y_label, Y_data_raw, _, _ = generateSplitDataset(train_files, config, error=True)

    pad = config['start_channel']
    Y_label = Y_label[:, pad:pad + config['n_channel'], :]
    Y_data_raw = Y_data_raw[:, pad:pad + config['n_channel'], :]

    # Test set (Load only two first samples to create validations set)
    # Disclamer: Do not load the test set completely, and more importantly do not use this as a mean of optimization
    # It is simply to have a sneak peak during training.
    x_out_tmp, y_out_label, y_out_raw, stretch_factor_out, _ = generateSplitDataset(test_files[:2], config, error=False)
    y_out_label = y_out_label[:, pad:pad + config['n_channel'], :]
    y_out_raw = y_out_raw[:, pad:pad + config['n_channel'], :]

    # Add first order-derivative
    x_out = np.zeros([x_out_tmp.shape[0], x_out_tmp.shape[1], 2 * x_out_tmp.shape[2]])
    x_out[:, :, :x_out_tmp.shape[2]] = x_out_tmp
    x_out[:, 1:, x_out_tmp.shape[2]:] = np.diff(x_out_tmp, n=1, axis=1)

    # Validation set
    # If you have access to a proper validation set please add it heer
    x_val = x_out
    y_val_label = y_out_label
    y_val_raw = y_out_raw

    # Training placeholder
    print('>>> Training:')
    stats_history = {'f1': [], 'precision': [], 'recall': [],
                     'f1_val': [], 'precision_val': [], 'recall_val': []}

    iter = 1
    while iter <= config['niter']:
        # Batch selection
        idx_batch = np.random.randint(0, len(Y_label), config['batch_size'])
        batch_x_tmp, batch_y, batch_y_series = \
            X_data[idx_batch, :, :], Y_label[idx_batch, :, :], Y_data_raw[idx_batch, :, :]

        # Add first order-derivative
        batch_x = np.zeros([batch_x_tmp.shape[0], batch_x_tmp.shape[1], 2 * batch_x_tmp.shape[2]])
        batch_x[:,:,:batch_x_tmp.shape[2]] = batch_x_tmp
        batch_x[:,1:, batch_x_tmp.shape[2]:] = np.diff(batch_x_tmp, n=1,axis=1)

        # Performance optimization
        softModel.optimize(sess, batch_x, batch_y, batch_y_series)

        if iter % config['dataset_update_frequency'] == 0 and iter > 0:
            # Update the dataset with new data (not used for the paper)
            print('Dataset Update...')
            x_new, y_new, y_new_raw, _, _ = generateSplitDataset(
                np.random.choice(train_files, config['dataset_update_size'], replace=False), config)
            y_new = y_new[:, pad:pad + config['n_channel'], :]
            y_new_raw = y_new_raw[:, pad:pad + config['n_channel'], :]

            idx_new = np.random.choice(X_data.shape[0], x_new.shape[0], replace=False)

            X_data[idx_new, :, :] = x_new
            Y_label[idx_new, :] = y_new
            Y_data_raw[idx_new, :, :] = y_new_raw

        if iter % config['show_frequency'] == 0:
            # Display training metrics
            acc, los, loss_dir, loss_ind, pp = softModel.infer(sess, batch_x, batch_y, batch_y_series)

            print("For iter ", iter)
            print("Accuracy ", acc)
            if config['Direct']:
                print("Loss ", np.round(los, 3), np.round(loss_dir, 3), np.round(loss_ind, 3))
            else:
                print("Loss ", np.round(los, 3))
            print("__________________")

            # Display (train) localization
            fig, stats = lp.localizationPlot(
                pp,
                batch_y_series, n_samples=20, dist_threshold=config['tolerence'], factor=config['downsamplingFactor'],
                bias=config['temporal_bias'])
            plt.savefig('plt/localization_in_' + config['extension'])
            plt.close()

            stats_history['f1'].append(stats['f1'])
            stats_history['precision'].append(stats['precision'])
            stats_history['recall'].append(stats['recall'])

            # Display (validation) localization
            pp = softModel.predict(sess, x_val)
            fig, stats_val = lp.localizationPlot(pp, y_val_raw, n_samples=20, dist_threshold=config['tolerence'],
                                                 factor=config['downsamplingFactor'], bias=config['temporal_bias'])
            plt.savefig('plt/localization_val_' + config['extension'])
            plt.close()

            stats_history['f1_val'].append(stats_val['f1'])
            stats_history['precision_val'].append(stats_val['precision'])
            stats_history['recall_val'].append(stats_val['recall'])

            # Display Loss & Performance
            softModel.performancePlot(stats_history)

            # Smoothed plot
            if config['Direct']:
                softModel.smoothPlot(sess, batch_x, batch_y, batch_y_series)

        iter += 1
