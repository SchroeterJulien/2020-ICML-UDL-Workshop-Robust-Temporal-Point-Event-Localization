import torch
import numpy as np
import scipy.ndimage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
smoothing_lambda = 5

# Smoothing Kernel
signal = np.zeros([21])  # 21
signal[len(signal) // 2] = 1
filt = scipy.ndimage.filters.gaussian_filter(signal, smoothing_lambda)
kernel = torch.FloatTensor(filt).float().unsqueeze_(0).unsqueeze_(0).to(device)

#### Classical Approach
def classicalLoss(pred, targets):

    # Transform targets
    target_distribution = torch.cuda.FloatTensor(targets.size()[0], targets.size()[1], pred.size()[2]).fill_(0)
    for ii in range(target_distribution.size()[2]):
        target_distribution[:, :, ii] = (targets == ii).type(torch.LongTensor)

    # Loss computation
    target_smooth = torch.conv1d(target_distribution.permute(0, 2, 1).reshape(-1, 1, 64), kernel,
                            padding=21 // 2).reshape(pred.size()[0], pred.size()[2], pred.size()[1]).permute(0,2,1)
    loss_smooth = torch.mean(torch.sum((pred - target_smooth) ** 2, 2))

    return loss_smooth, 0*loss_smooth


#### Our Model
non_normalized_kernel = torch.FloatTensor(filt / np.max(filt)).float().unsqueeze_(0).unsqueeze_(0).to(device)
# Note: a standard Gaussian instead of that scaled one could be used without any issue.
# The learning rate however would need to be modified accordingly
def SoftLocClassic(pred, targets):

    # Transform targets
    target_distribution = torch.cuda.FloatTensor(targets.size()[0], targets.size()[1], pred.size()[2]).fill_(0)
    for ii in range(target_distribution.size()[2]):
        target_distribution[:, :, ii] = (targets == ii).type(torch.LongTensor)

    # Apply smoothing
    x_smooth = torch.conv1d(pred.permute(0, 2, 1).reshape(-1, 1, pred.size()[1]), non_normalized_kernel, padding=21 // 2).reshape(
        pred.size()[0], pred.size()[2], pred.size()[1]).permute(0, 2, 1)
    target_smooth = torch.conv1d(target_distribution.permute(0, 2, 1).reshape(-1, 1, 64), non_normalized_kernel,
                                 padding=21 // 2).reshape(pred.size()[0], pred.size()[2], pred.size()[1]).permute(0, 2,
                                                                                                                  1)
    # Compute Loss
    loss_smooth = torch.sum(torch.sum((x_smooth - target_smooth) ** 2, 2)) / pred.shape[0] / (smoothing_lambda * np.sqrt(2 * np.pi))
    loss_count = CountingLoss(pred, targets)

    return loss_smooth, loss_count


def CountingLoss(pred, targets):
    loss_count = 0

    contribution = torch.unbind(pred[:, :, :], 1)

    max_occurence = 3
    count_prediction = torch.cuda.FloatTensor(pred.size()[0], pred.size()[2], max_occurence).fill_(0)
    count_prediction[:, :, 0] = 1  # (batch x class x max_occ)
    for increment in contribution:
        mass_movement = (count_prediction * increment.unsqueeze(2))[:, :, :max_occurence - 1]
        move = - torch.cat([mass_movement,
                            torch.cuda.FloatTensor(count_prediction.size()[0], count_prediction.size()[1], 1).fill_(
                                0)], axis=2) \
               + torch.cat(
            [torch.cuda.FloatTensor(count_prediction.size()[0], count_prediction.size()[1], 1).fill_(0),
             mass_movement], axis=2)

        count_prediction = count_prediction + move

    # Compute Target Count Distributions
    target_distribution = torch.cuda.FloatTensor(count_prediction.size()[0], count_prediction.size()[1],
                                                 max_occurence).fill_(0)
    for ii in range(targets.size()[0]):
        for kk in range(pred.size()[2]):
            target_distribution[ii, kk, torch.sum(targets[ii] == kk).type(torch.LongTensor)] = 1

    # Compare Estimated and Target distributions
    loss_count -= torch.sum(torch.sum(torch.log(count_prediction + 1e-12) * target_distribution, [1, 2]))

    return loss_count
