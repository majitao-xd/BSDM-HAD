import random
import argparse
import numpy as np
import torch
import os
from scipy.io import loadmat, savemat

from models import get_beta_schedule, GaussianDiffusion, Network
from noise_fn import generate_pseudo_background_noise
from anomaly_detector import run_rx
from utils import normalize, false_alarm_rate, channel_align


def get_args() -> argparse.Namespace:
    """parser args"""
    parser = argparse.ArgumentParser(description='BSDM training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset options
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='Folder to datasets')
    parser.add_argument('--train_data', type=str, default='SanDiego', help='Name of train set.')
    parser.add_argument('--test_data', type=str, default='None', help='Name of test set.')

    # The path of files to save
    parser.add_argument('--save_result', action='store_true', default=False, help='Save background suppression result.')
    parser.add_argument('--result_save_dir', type=str, default='./results/', help='Folder to save background suppression results.')

    # Training options
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001, help='The Initial Learning Rate.')
    parser.add_argument('--diffusion_mode', type=str, default='gamma', choices=['gamma', 'alpha'], help='Methods of noise diffusion.')
    parser.add_argument('--T', type=int, default=1000, help='Maximum time step.')
    parser.add_argument('--t', type=int, default=500, help='Time step.')
    parser.add_argument('--K', type=int, default=10, help='Number of inference.')
    parser.add_argument('--RX', action='store_true', default=True, help='Verification with RX detector.')

    # Model options
    parser.add_argument('--pre_embed_dim', type=int, default=128, help='Number of pre-embedded channels of de-noising network.')
    parser.add_argument('--hidden_dim', type=list, default=[200, 100, 50], help='Number of hidden channels of de-noising network.')

    # Random seed
    parser.add_argument('--seed', type=int, default=None, help='Manual seed.')

    # Device to use
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Device to use.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use.')

    return parser.parse_args()


def set_seed(seed=None):
    if seed is None:
        set_seed(random.randint(0, 2 ** 30))
    else:
        print('===>>> Seed = {} <<<==='.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def training(d_model, model, x, opt, scheduler):
    model.train()
    for e in range(args.epochs):
        loss, _, _, _ = d_model.p_loss(model, x)
        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()
        print('Epoch-{}: loss = {}'.format(e+1, loss.item()))


def test(d_model, model, x, train_channels, k):
    x_new = channel_align(x, train_channels)
    x = np.concatenate(x_new, axis=-1)
    n = len(x_new)

    for i in range(n):
        pre_x_0 = torch.tensor(np.expand_dims(np.transpose(x_new[i], [2, 0, 1]), 0)).to(device)
        for _ in range(k):
            _, pre_x_0 = d_model.test(model, pre_x_0)
        pre_x_0 = np.transpose(pre_x_0.squeeze().cpu().numpy(), [1, 2, 0])
        x_new[i] = pre_x_0

    x_new = np.concatenate(x_new, axis=-1)

    return x_new, x


if __name__ == '__main__':
    # ger args
    args = get_args()

    # set device
    device = args.device
    if device == 'gpu':
        device = args.gpu_id

    # set seed
    set_seed(args.seed)

    # load data
    train_data_path = args.data_dir + args.train_data + '.mat'
    train_set = loadmat(train_data_path)
    train_data = train_set['data'].astype(np.float32)
    train_map = train_set['map']

    # normalize and translation to tensor
    x = normalize(train_data)
    x = torch.tensor(np.expand_dims(np.transpose(x, [2, 0, 1]), 0)).to(device)

    # build model
    beta = get_beta_schedule(args.T)
    noise = generate_pseudo_background_noise(x, t=args.t)

    model = Network({'in_channels': x.shape[1], 'pre_embed_dim': args.pre_embed_dim, 'hidden_dim': args.hidden_dim}).to(device)
    d_model = GaussianDiffusion(beta, noise=noise, diffusion_mode='gamma', t=args.t).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, args.lr/10)

    # training
    training(d_model, model, x, opt, scheduler)

    # test
    train_channels = x.shape[1]

    if args.test_data == 'None':
        test_data_path = train_data_path
    else:
        test_data_path = args.data_dir + args.test_data + '.mat'
    test_set = loadmat(test_data_path)
    test_data = test_set['data'].astype(np.float32)
    test_map = test_set['map']

    x = normalize(test_data)

    result, x = test(d_model, model, x, train_channels, args.K)

    # verification
    if args.RX:
        file_name = os.path.splitext(os.path.split(test_data_path)[-1])[0]
        rx_result = run_rx(x)
        auc, fpr = false_alarm_rate(test_map.reshape([-1]), rx_result.reshape([-1]))
        print('RX_detector | ' + file_name + ' | AUC: {}, FPR: {}'.format(auc, fpr))

        if args.save_result:
            save_path = args.result_save_dir + 'result-t{}-K0'.format(args.t) + file_name + '.mat'
            savemat(save_path, {'res': rx_result, 'map': test_map})

        rx_result_bs = run_rx(result)
        auc, fpr = false_alarm_rate(test_map.reshape([-1]), rx_result_bs.reshape([-1]))
        print('RX_detector | with BSDM | ' + file_name + ' | AUC: {}, FPR: {}'.format(auc, fpr))

        if args.save_result:
            save_path = args.result_save_dir + 'result-t{}-K{}'.format(args.t, args.K) + file_name + '.mat'
            savemat(save_path, {'res': rx_result, 'map': test_map})

