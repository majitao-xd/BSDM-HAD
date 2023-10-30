import torch
import numpy as np
from torch import nn


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        # parameters
        self.in_channels = config['in_channels']
        self.pre_embed_dim = config['pre_embed_dim']
        self.hidden_dim = config['hidden_dim']

        self.time_embed_dim = self.pre_embed_dim * 4

        # modules
        self.time_embedding = nn.Sequential(
            PositionalEmbedding(self.pre_embed_dim),
            nn.Linear(self.pre_embed_dim, self.time_embed_dim),
            nn.Tanh(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.shift_embedding = nn.Sequential(
            nn.Linear(2 * self.in_channels, self.pre_embed_dim * 4),
            nn.Tanh(),
            nn.Linear(self.pre_embed_dim * 4, self.pre_embed_dim),
        )

        self.pre_embed_layer = nn.Linear(self.in_channels, self.pre_embed_dim)

        self.down = nn.ModuleList(
            [res_layer(self.pre_embed_dim, self.time_embed_dim, self.hidden_dim[0], self.hidden_dim[1]),
             res_layer(self.hidden_dim[1], self.time_embed_dim, self.hidden_dim[0], self.hidden_dim[2]),]
        )

        self.middle = nn.ModuleList(
            [res_layer(self.hidden_dim[2], self.time_embed_dim, self.hidden_dim[1], self.hidden_dim[2]),
            res_layer(self.hidden_dim[2], self.time_embed_dim, self.hidden_dim[1], self.hidden_dim[2]),]
        )

        self.up = nn.ModuleList(
            [res_layer(self.hidden_dim[2], self.time_embed_dim, self.hidden_dim[0], self.hidden_dim[1]),
            res_layer(self.hidden_dim[1], self.time_embed_dim, self.hidden_dim[0], self.pre_embed_dim),]
        )

        self.out_layer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.pre_embed_dim, self.in_channels)
        )

    def forward(self, x, t, distribution_shift=True):
        B, C, M, N = x.shape
        x = x.reshape([B, C, M*N]).transpose(1, 2)

        if distribution_shift:
            shift = torch.cat([torch.mean(x, dim=-1, keepdim=True).repeat(1, 1, self.in_channels), torch.std(x, dim=-1, keepdim=True).repeat(1, 1, self.in_channels)], dim=-1).to(x.device)
            shift = self.shift_embedding(shift)

        else:
            shift = 0

        time_embed = self.time_embedding(t)
        h = self.pre_embed_layer(x) + shift
        for _, layer in enumerate(self.down):
            h = layer(h, time_embed)
        for _, layer in enumerate(self.middle):
            h = layer(h, time_embed)
        for _, layer in enumerate(self.up):
            h = layer(h, time_embed)
        h = self.out_layer(h)

        h = h.transpose(1, 2).reshape([B, C, M, N])

        return h


class res_layer(nn.Module):
    def __init__(self, in_channels, embed_dim=None, num_hidden = 200, out_channels=None):
        super(res_layer, self).__init__()
        # parameters
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_hidden = num_hidden
        self.out_channels = out_channels or in_channels

        # modules
        self.feedforward = nn.Sequential(
            nn.LayerNorm(self.in_channels),
            nn.Tanh(),
            nn.Linear(in_features=self.in_channels, out_features=self.num_hidden),
            nn.Tanh(),
            nn.Linear(in_features=self.num_hidden, out_features=self.out_channels)
        )
        self.time_embed_layers = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.embed_dim, self.out_channels)
        )
        self.out_layer = nn.Sequential(
            nn.LayerNorm(self.out_channels),
            nn.Tanh(),
            nn.Linear(self.out_channels, self.out_channels)
        )
        if self.in_channels == self.out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x, time_embed):
        emb_out = self.time_embed_layers(time_embed)
        h = self.feedforward(x)
        h = h + emb_out
        h = self.out_layer(h)
        skip_h = self.skip_connection(x)

        return h + skip_h


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
