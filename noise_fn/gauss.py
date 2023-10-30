import torch


def generate_pseudo_background_noise(x, t=1, mean=None, std=None):
    if mean is None and std is None:
        mean = torch.mean(x)
        std = torch.std(x)
    noise = torch.normal(mean=mean, std=std, size=x.shape).to(x.device)
    print('===>>> noise mean={}, std={} <<<==='.format(mean, std))

    return noise