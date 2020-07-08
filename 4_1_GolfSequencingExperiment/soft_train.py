import matplotlib
matplotlib.use('Agg')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt

from dataloader import GolfDB, Normalize, ToTensor
from model import EventDetector
from util import *

import SmoothedLosses as sloss
from Math.NadarayaWatson import *


import sys
argv = sys.argv[1:]
if len(argv)>0:
    noise_level = float(argv[0])
    print('>>> Noise: ' + str(noise_level))
else:
    noise_level = 0

# Split
if len(argv)>1:
    split = int(argv[1])
    print('>>> split: ' + str(split))
else:
    split = 1

if len(argv) > 2:
    bool_classical_loss = argv[2].lower() == 'true'
else:
    bool_classical_loss = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if bool_classical_loss:
    version_name = 'classical_' + str(split) + '_' + str(noise_level)
    config = {'Adam': True, 'iterations': 10000, 'show_frequency': 100, 'learning_rate': 0.0002}
else:
    version_name = 'softloc_' + str(split) + '_' + str(noise_level)
    config = {'Adam': True, 'iterations': 10000, 'show_frequency':100, 'alpha':0.005, 'learning_rate':0.0002, 'start_converging': 7000}

print(version_name)

#version_name = "SwingNet_bench"
loss_window = {'loss': np.zeros([25]), 'soft': np.zeros([25]), 'count': np.zeros([25])}
list_loss = {'loss':[], 'soft':[], 'count':[]}


if __name__ == '__main__':

    # training configuration (From McNally et al.)
    iterations = config['iterations']
    it_save = 10000  # save model every 10000 iterations
    n_cpu = 6
    seq_length = 64
    bs = 22  # batch size
    k = 10  # frozen layers

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    model.train()
    model.to(device)

    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=True,
                     noise_level=noise_level)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    if config['Adam']:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate']) #todo(default):0.001
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate']) #todo(default):0.001

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    while i < iterations:
        for sample in data_loader:
            images, labels = sample['images'].to(device), sample['labels'].to(device)
            logits = model(images)

            if bool_classical_loss:
                preds = torch.reshape(torch.sigmoid(logits), [bs, seq_length, 9])[:, :, :-1]

                loss_smooth, loss_count = sloss.classicalLoss(preds, labels)
                loss = loss_smooth

            else:
                preds = torch.reshape(torch.sigmoid(logits),[bs, seq_length, 9])[:,:,:-1]
                loss_smooth, loss_count = sloss.SoftLocClassic(preds, labels)
                loss = loss_smooth + min(max(i-config['start_converging'],0)/2000,1) * config['alpha'] * loss_count

            loss_window['loss'][1:] = loss_window['loss'][:-1]
            loss_window['loss'][0] = loss.cpu().data.numpy()
            list_loss['loss'].append(np.median(loss_window['loss'][loss_window['loss'] != 0]))

            loss_window['soft'][1:] = loss_window['soft'][:-1]
            loss_window['soft'][0] = loss_smooth.cpu().data.numpy()
            list_loss['soft'].append(np.median(loss_window['soft'][loss_window['soft'] != 0]))

            loss_window['count'][1:] = loss_window['count'][:-1]
            loss_window['count'][0] = config['alpha'] * loss_count.cpu().data.numpy()
            list_loss['count'].append(np.median(loss_window['count'][loss_window['count'] != 0]))


            if i % config['show_frequency'] == 0:
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.plot(np.log(list_loss['loss']), 'k', alpha=0.5, linewidth=1)
                plt.plot(1 * np.arange(1, 1 + len(list_loss['loss'])),
                         GaussKernel(np.arange(1, 1 + len(list_loss['loss'])),
                                     np.arange(1, 1 + len(list_loss['loss'])),
                                     np.log(np.array(list_loss['loss'])), config['show_frequency']),
                         'k', linewidth=2)

                plt.plot(np.log(list_loss['soft']), 'b', alpha=0.5, linewidth=1)
                plt.plot(1 * np.arange(1, 1 + len(list_loss['soft'])),
                         GaussKernel(np.arange(1, 1 + len(list_loss['soft'])),
                                     np.arange(1, 1 + len(list_loss['soft'])),
                                     np.log(np.array(list_loss['soft'])), config['show_frequency']),
                         'b', linewidth=2)

                plt.plot(np.log(np.array(list_loss['count'])), 'orange', alpha=0.5, linewidth=1)
                plt.plot(1 * np.arange(1, 1 + len(list_loss['count'])),
                         GaussKernel(np.arange(1, 1 + len(list_loss['count'])),
                                     np.arange(1, 1 + len(list_loss['count'])),
                                     np.log(np.array(list_loss['count'])), config['show_frequency']),
                         'orange', linewidth=2)

                plt.ylim([np.log(min(np.min(list_loss['loss']), np.min(list_loss['soft']), np.min(list_loss['count']))), np.log(list_loss['loss'][min(500,max(0,i-100))])])

                plt.subplot(1, 2, 2)
                plt.hist(preds.reshape(-1).cpu().data.numpy())
                plt.ylim([0,50])

                plt.savefig('loss_' + version_name + '.png')
                plt.close('all')


            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(i, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/{}_{}.pth.tar'.format(version_name, i))
            if i == iterations:
                break

    # Run evaluation
    import os
    os.system("python eval.py " + version_name)





