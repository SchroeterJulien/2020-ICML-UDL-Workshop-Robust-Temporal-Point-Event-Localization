from dataloader import GolfDB, Normalize, ToTensor
from model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import sys
argv = sys.argv[1:]
if len(argv)>0:
    noise_level = float(argv[0])
    print('>>> step: ' + str(noise_level))
else:
    noise_level = 0

if len(argv)>1:
    split = int(argv[1])
    print('>>> split: ' + str(split))
else:
    split = 1





version_name = 'original_' + str(split) + '_' + str(noise_level)
print(version_name)
if __name__ == '__main__':

    # training configuration
    iterations = 10000
    it_save = 10000  # save model every 100 iterations
    n_cpu = 8
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
    model.cuda()

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
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    while i < iterations:

        for sample in data_loader:


            images, labels = sample['images'].cuda(), sample['labels'].cuda()

            logits = model(images)
            labels = labels.view(bs*seq_length)
            loss = criterion(logits, labels)
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
