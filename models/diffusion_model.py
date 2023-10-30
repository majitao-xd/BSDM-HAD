import torch
from torch import nn
import numpy as np


def get_beta_schedule(num_diffusion_steps=1000, name='linear'):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas


def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 betas,
                 noise,
                 diffusion_mode='alpha', # alpha, gamma
                 t=None,
                 ):
        super(GaussianDiffusion, self).__init__()

        if t is None:
            self.t = 0
        else:
            self.t = t

        self.noise = noise
        self.diffusion_mode = diffusion_mode

        self.num_timesteps = len(betas)

        alphas = 1. - betas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.phi_t = np.sqrt(np.flip(np.cumprod(np.flip(alphas[0:t]), axis=0)) * self.betas[0:t])
        self.gamma_t = np.sum(self.phi_t)

    def predict_x_0_from_eps(self, x_t, t, eps):
        if self.diffusion_mode == 'gamma':
            return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * (x_t - self.gamma_t * eps)
        elif self.diffusion_mode == 'alpha':
            return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)
        else:
            return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)

    def sample_q(self, x_0, t, noise):
        k1 = extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device)
        if self.diffusion_mode == 'gamma':
            x_t = k1 * x_0 + self.gamma_t * noise
        elif self.diffusion_mode == 'alpha':
            k2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device)
            x_t = k1 * x_0 + k2 * noise
        else:
            k2 = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device)
            x_t = k1 * x_0 + k2 * noise

        return x_t

    def p_loss(self, model, x_0):
        t = torch.randint(1, self.num_timesteps, (x_0.shape[0],), device=x_0.device)
        if self.t != 0:
            t = t * 0 + self.t

        x_t = self.sample_q(x_0, t, self.noise)
        estimate_noise = model(x_t, t)
        estimate_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise)

        loss = torch.mean((estimate_noise - self.noise).square())

        return loss, x_t, estimate_noise, estimate_x_0

    @torch.no_grad()
    def test(self, model, x):
        t = torch.randint(1, self.num_timesteps, (x.shape[0],), device=x.device)
        if self.t != 0:
            t = t * 0 + self.t

        estimate_noise = model(x, t)
        pred_x_0 = self.predict_x_0_from_eps(x, t, estimate_noise)

        return estimate_noise, pred_x_0