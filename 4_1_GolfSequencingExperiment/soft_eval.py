import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import os

from dataloader import GolfDB, ToTensor, Normalize
from model import EventDetector
from util import correct_preds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import sys
argv = sys.argv[1:]
if len(argv)>0:
    noise_level = float(argv[0])
    print('>>> step: ' + str(noise_level))
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


if bool_classical_loss:
    version_name = 'classical_' + str(split) + '_' + str(noise_level)
else:
    version_name = 'softloc_' + str(split) + '_' + str(noise_level)

print(version_name)

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

            logits = model(image_batch.to(device))
            if batch == 0:
                probs = F.sigmoid(logits.data).cpu().numpy()
            else:
                probs = np.append(probs, F.sigmoid(logits.data).cpu().numpy(), 0)
            batch += 1
        _, _, _, _, c = correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)
        correct.append(c)
    PCE = np.mean(correct)
    return PCE


if __name__ == '__main__':

    seq_length = 64
    n_cpu = 6

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load('models/' + version_name + '_10000.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    PCE = eval(model, split, seq_length, n_cpu, True)
    print('Average PCE: {}'.format(PCE))

    if not os.path.exists('results'):
        os.mkdir('results')

    if bool_classical_loss:
        output_file = 'results/classical.txt'
    else:
        output_file = 'results/softloc.txt'

    with open(output_file, 'a') as f:
        f.write(version_name)
        f.write(",%s" % str(noise_level))
        f.write(",%s" % str(PCE))
        f.write("\n")

