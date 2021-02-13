import math
import torch
import torch.nn.functional as F
from torch import nn
from distributions import DiscretizedLogistic

# for Cifar10
IMAGE_SIZE     = 32
IMAGE_CHANNELS = 3

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.zeros([]))

    def forward(self, x, **kwargs):
        return x + self.alpha * self.fn(x, **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), 
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfAttention(nn.Module):

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout)

    def forward(self, x):
        return self.attn(x, x, x)[0]


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(nn.Sequential(nn.LayerNorm(dim), SelfAttention(dim, heads, dropout))),
                Residual(FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, patch_size, num_hiddens, dim, depth, heads, mlp_dim, dropout=0, emb_dropout=0):
        super().__init__()
        assert IMAGE_SIZE % patch_size == 0
        self.num_patches = (IMAGE_SIZE // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_dim = IMAGE_CHANNELS * patch_size**2
        self.dim = dim

        self.pos_embedding      = nn.Parameter(torch.zeros(self.num_patches + 1, 1, dim))
        self.patch_to_embedding = nn.Sequential(nn.Linear(self.patch_dim, dim), nn.LayerNorm(dim))
        self.dropout            = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_hiddens*2)
        )

    def forward(self, img):
        p = self.patch_size
        N, C, H, W = img.size()

        x = img.view(N, C, H//p, p, W//p, p).permute(2, 4, 0, 3, 5, 1).reshape(self.num_patches, N, self.patch_dim)
        x = self.patch_to_embedding(x)

        x = torch.cat([x, x.new_zeros(1, N, self.dim)], dim=0)
        x = self.dropout(x + self.pos_embedding * math.sqrt(self.dim))

        x = self.transformer(x)

        x = self.mlp_head(x[-1])
        loc, scale = torch.chunk(x, 2, dim=-1)
        scale = F.softplus(scale) + 1e-6

        return torch.distributions.Normal(loc, scale)

class ViTDecoder(nn.Module):
    def __init__(self, patch_size, num_hiddens, dim, depth, heads, mlp_dim, dropout=0, emb_dropout=0):
        super().__init__()
        assert IMAGE_SIZE % patch_size == 0
        self.num_patches = (IMAGE_SIZE // patch_size) ** 2
        self.patch_dim = IMAGE_CHANNELS * patch_size ** 2
        self.dim = dim
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.zeros(self.num_patches, 1, dim))
        self.patch_to_img  = nn.Linear(dim, self.patch_dim*2)
        self.dropout       = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.mlp_head = nn.Sequential(nn.Linear(num_hiddens, dim, bias=False), nn.LayerNorm(dim))

    def forward(self, z):
        N, H = z.size()
        p = self.patch_size

        x = self.mlp_head(z).unsqueeze(0) # (1, N, dim)
        # broadcast x
        x = self.dropout(x + self.pos_embedding * math.sqrt(self.dim))

        x = self.transformer(x)

        x = self.patch_to_img(x) # (H//p)*(W//p), N, p*p*CH
        SEQ, N, CH = x.size()
        L = int(math.sqrt(SEQ))
        x = x.view(L, L, N, p, p, CH//p//p)
        x = x.permute(2, 5, 0, 3, 1, 4).reshape(N, CH//p//p, L*p, L*p)
        loc, scale = torch.chunk(x, 2, dim=1)

        return DiscretizedLogistic(loc, F.softplus(scale) + 1e-6)


class ViTVAE(nn.Module):

    def __init__(self, patch_size, num_hiddens, dim, depth, heads, mlp_dim, dropout=0, emb_dropout=0):
        super().__init__()

        self.encoder = ViTEncoder(patch_size, num_hiddens, dim, depth, heads, mlp_dim, dropout, emb_dropout)
        self.decoder = ViTDecoder(patch_size, num_hiddens, dim, depth, heads, mlp_dim, dropout, emb_dropout)

    def forward(self, x, alpha=1./24.):

        dist = self.encoder(x)
        z = dist.rsample()
        rx = self.decoder(z)

        prior = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))

        reconstruction_error = -rx.log_prob(x).mean()
        regularization_parameter = torch.distributions.kl.kl_divergence(prior, dist).mean()
        loss = reconstruction_error + alpha * regularization_parameter

        losses = dict()
        variables = dict()
        losses["loss"]                     = loss
        losses["reconstruction_error"]     = reconstruction_error
        losses["regularization_parameter"] = regularization_parameter 
        variables["rx"]                       = rx
        variables["z"]                        = z
        variables["dist"]                     = dist

        return losses, variables
