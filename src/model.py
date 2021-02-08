import math
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
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
                Residual(nn.Sequential(nn.LayerNorm(dim), FeedForward(dim, mlp_dim, dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, image_size, patch_size, num_hiddens, dim, depth, heads, mlp_dim, channels=3, dropout=0, emb_dropout=0):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_dim = channels * patch_size**2
        self.dim = dim

        self.pos_embedding      = nn.Parameter(torch.randn(self.num_patches + 1, 1, dim))
        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)
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
        x = self.dropout(x + self.pos_embedding)

        x = self.transformer(x)

        x = self.mlp_head(x[-1])
        loc, scale = torch.chunk(x, 2, dim=-1)
        scale = F.softplus(scale) + 1e-6

        return torch.distributions.Normal(loc, scale)

class ViTDecoder(nn.Module):
    def __init__(self, image_size, patch_size, num_hiddens, dim, depth, heads, mlp_dim, channels=3, dropout=0, emb_dropout=0):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = channels * patch_size ** 2
        self.dim = dim
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(self.num_patches + 1, 1, dim))
        self.patch_to_img  = nn.Linear(dim, self.patch_dim)
        self.dropout       = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.mlp_head = nn.Linear(num_hiddens, dim, bias=False) # pos_embedding instead of bias

    def forward(self, z):
        N, H = z.size()
        p = self.patch_size

        x = self.mlp_head(z).unsqueeze(0) # (1, N, dim)
        x = torch.cat([x.new_zeros([self.num_patches, N, self.dim]), x], dim=0)
        x = self.dropout(x + self.pos_embedding)

        x = self.transformer(x)

        x = self.patch_to_img(x[:-1]) # (H//p)*(W//p), N, p*p*CH
        SEQ, N, CH = x.size()
        L = int(math.sqrt(SEQ))
        x = x.view(L, L, N, p, p, CH//p//p)
        x = x.permute(2, 5, 0, 3, 1, 4).reshape(N, CH//p//p, L*p, L*p)

        return torch.distributions.Normal(x, 1.0)


class ViTVAE(nn.Module):

    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0, emb_dropout=0):
        super().__init__()

        self.encoder = ViTEncoder(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout)
        self.decoder = ViTDecoder(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels, dropout, emb_dropout)

    def forward(self, x, alpha=1.0):

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
