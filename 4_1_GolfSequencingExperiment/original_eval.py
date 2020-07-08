import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

from dataloader import GolfDB, ToTensor, Normalize
from model import EventDetector
from util import correct_preds


import sys
argv = sys.argv[1:]
if len(argv)>0:
    noise_level = float(argv[0])
    print('>>> step: ' + str(noise_level))
else:
    noise_level = 0

if len(argv)>1:
    split = int(argv[1])
    print('>>> split: ' + str(noise_level))
else:
    split = 1


version_name = 'original_' + str(split) + '_' + str(noise_level)


def eval(model, split, seq_length, n_cpu, disp):
    dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)


    correct = []
    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']

        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        _, _, _, _, c = correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)
        correct.append(c)
    PCE = np.mean(correct)
    return PCE


if __name__ == '__main__':

    seq_length = 64
    n_cpu = 8

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load('models/' + version_name + '_10000.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, True)
    print('Average PCE: {}'.format(PCE))


    if not os.path.exists('results'):
        os.mkdir('results')

    with open('results/original_results.txt', 'a') as f:
        f.write(version_name)
        f.write(",%s" % str(noise_level))
        f.write(",%s" % str(PCE))
        f.write("\n")


