import numpy as np
import matplotlib.pyplot as plt
import math

def false_alarm_rate(target, predicted, show_fig: bool = False):
    """
    Calculate AUC and false alarm auc
    :param target: [n,1]
    :param predicted: [n,1]
    :param show_fig: default false
    :return: PD_PF_AUC, PF_tau_AUC
    """
    target = np.array(target)
    predicted = np.array(predicted)
    if target.shape != predicted.shape:
        assert False, 'Wrong shape!'
    target = (target - target.min()) / (target.max() - target.min())
    predicted = (predicted - predicted.min()) / (predicted.max() - predicted.min())
    anomaly_map = target
    normal_map = 1 - target

    num = 30000
    taus = np.linspace(0, predicted.max(), num=num)
    PF = np.zeros([num, 1])
    PD = np.zeros([num, 1])

    for index in range(num):
        tau = taus[index]
        anomaly_map_1 = np.double(predicted >= tau)
        PF[index] = np.sum(anomaly_map_1 * normal_map) / np.sum(normal_map)
        PD[index] = np.sum(anomaly_map_1 * anomaly_map) / np.sum(anomaly_map)

    if show_fig:
        plt.figure(1)
        plt.plot(PF, PD)
        plt.figure(2)
        plt.plot(taus, PD)
        plt.figure(3)
        plt.plot(taus, PF)
        plt.show()

    PD_PF_auc = np.sum((PF[0:num - 1, :] - PF[1:num, :]) * (PD[1:num] + PD[0:num - 1]) / 2)
    PF_tau_auc = np.trapz(PF.squeeze(), taus.squeeze())

    return PD_PF_auc, PF_tau_auc


def normalize(data):
    """
    normalize data to [0, 1]
    :param data: input data
    :return: data in [0, 1]
    """
    data = (data - data.min()) / (data.max() - data.min())

    return data


def channel_align(x, target_channels):
    """
    Set the number of channels of x to target_channels.
    :param x: tensor
    :param target_channels: int
    :return: channel aligned x
    """
    channels = x.shape[-1]

    if channels == target_channels:
        return [x]

    x_new = []
    if channels / target_channels < 2:
        if channels < target_channels:
            right = int((target_channels- channels) / 2)
            left = int(target_channels - channels - right)
            x = np.pad(x, ((0, 0), (0, 0), (left, right)), mode='reflect')
        elif channels > target_channels:
            c = np.sort(np.random.choice(channels, target_channels, replace=False))
            x = x[:,:,c]
        x_new.append(x)

    else:
        n = math.floor(channels / target_channels)
        c = np.sort(np.random.choice(channels, target_channels * n, replace=False))
        for i in range(n):
            x_new.append(x[:, :, c[target_channels * i:target_channels * (i+1)]])

    return x_new

