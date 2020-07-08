# Configuration file

# Almost all parameters are fixed. However, all these parameters allow a flexible application of the model on a wide
# array of problems.

import numpy as np
import os.path

def load_configurations(extension="", start=None, noise=0, oneSided=False):

    config = {}
    if start is not None:
        if oneSided:
            config['extension'] = 'soft_oneSided_' + str(noise) + 'ms'
        else:
            config['extension'] = 'soft_model_' + str(noise) + 'ms'
    else:
        config['extension'] = 'generic_model'

    if os.path.isfile('models/' + extension + '.npy'):
        # If the model already exists simply load configurations from backup
        print('Load from backup')
        config = np.load('models/' + config['extension'] + '.npy').item()

    else:
        # Else load new configurations

        config['oneSided_smoothing'] = oneSided

        # Extension Name
        config['bidirectional'] = False

        if start is not None:
            config['start_channel'] = int(start)
            config['label_noise'] = int(noise) / 1000
        else:
            config['start_channel'] = 30
            config['label_noise'] = 0.150

        # Number of channels (~first 5 and last 5 are discarded since almost no occurence)
        config['n_channel'] = 70

        # Number of available cores for the computation
        config['cores'] = 40
        # GPU device
        config['GPU_Device'] = "0"

        # Augmentation factor (Caution with the memory!)
        config['augmentation_factor'] = 7

        # 50ms tolerence as in Hawthrone et al.
        config['tolerence'] = 10 // config['downsamplingFactor']
        config['temporal_bias'] = 0

        # Smooth settings
        config['start_processing'] = 100000
        config['weight_indirect'] = 0.2
        config['smoothing_width'] = 20

        # Run settings
        config['niter'] = 160000

        # Network settings
        config['num_units'] = 128
        config['hidden_size'] = 96
        config['n_filters'] = [32, 32, 64, 64, 64, 64,  64, 128, 128]

        # Save and display settings
        config['show_frequency'] = 10000
        config['save_frequency'] = 10000

        # Learning settings
        config['clipping_ratio'] = 10
        config['learning_rate'] = 0.0003
        config['batch_size'] = 32
        config['bool_gradient_clipping'] = True

        # Dataset settings
        config['dataset_size'] = 48
        config['dataset_update_size'] = 8
        config['dataset_update_frequency'] = 1e12
        config['update_start'] = 0

        # Dataset constants
        config['time_steps'] = 400
        config['n_filter'] = 384 * 2
        config['max_occurence'] = 40 #0,....,x-2, rest (so not exactly max occurence)

        # Dataset options
        config['ensembling_factor'] = 0.04 * config['augmentation_factor']

        # Dataset creation settings
        config['signal_pad'] = 0.0
        config['signal_noise'] = 0
        config['split_length'] = 1.5  # 1.5
        config['split_step'] = 1.5  # 1.5

        # Spectrogram settings
        config['spectral_size'] = 550

        # Detection threshold
        config['trigger_threshold'] = 0.2

        # Settings for other applications
        config['downsamplingFactor'] = 1 # Allow to add some temporal downsampling (beta-version)
        config['Direct'] = True # Turn the model into model presented in [2]

        # Save settings
        np.save('models/' + config['extension'] + '.npy', config)

    return config
