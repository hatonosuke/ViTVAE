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

class VAEBase(nn.Module):

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


class ViTVAE(VAEBase):

    def __init__(self, patch_size, num_hiddens, dim, depth, heads, mlp_dim, dropout=0, emb_dropout=0):
        super().__init__()

        self.encoder = ViTEncoder(patch_size, num_hiddens, dim, depth, heads, mlp_dim, dropout, emb_dropout)
        self.decoder = ViTDecoder(patch_size, num_hiddens, dim, depth, heads, mlp_dim, dropout, emb_dropout)


def group_norm16(dim):
    return nn.GroupNorm(dim//16, dim)

class Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        if stride == 1:
            assert in_channels == out_channels
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(nn.AvgPool2d(stride, stride), nn.Conv2d(in_channels, out_channels, 1, bias=False), group_norm16(out_channels))
        self.branch = nn.Sequential(
            group_norm16(in_channels), nn.ReLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False),
            group_norm16(out_channels), nn.ReLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.alpha = nn.Parameter(torch.zeros([]))

    def forward(self, x):
        return self.skip(x) + self.alpha * self.branch(x)


class ConvEncoder(nn.Module):

    def __init__(self, num_hiddens, base_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, base_dim, 3, padding=1, bias=False), nn.GroupNorm(base_dim//16, base_dim), # 32x32
            Block(base_dim,   base_dim), # 32x32
            Block(base_dim,   base_dim*2, stride=2), # 16x16
            Block(base_dim*2, base_dim*4, stride=2), #  8x8
            Block(base_dim*4, base_dim*8, stride=2), #  4x4
        )
        self.head = nn.Sequential(group_norm16(base_dim*8*4*4), nn.Linear(base_dim*8*4*4, num_hiddens*2))

    def forward(self, x):
        features = self.features(x)
        N, C, H, W = features.size()
        y = self.head(features.view(N, -1))

        loc, scale = torch.chunk(y, 2, dim=1)

        return torch.distributions.Normal(loc, F.softplus(scale) + 1e-6)


class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*2*2, 3, padding=1, bias=False), 
                nn.PixelShuffle(2), group_norm16(out_channels), nn.ReLU(), 
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                group_norm16(out_channels), nn.ReLU()
            )
        self.last_conv = nn.Conv2d(out_channels, IMAGE_CHANNELS*2, 1)

    def forward(self, x):
        y = self.block(x)
        img = F.interpolate(self.last_conv(y), size=(IMAGE_SIZE, IMAGE_SIZE), mode='nearest')

        return y, img


class ConvDecoder(nn.Module):

    def __init__(self, num_hiddens, base_dim):
        super().__init__()
        self.base_dim = base_dim

        self.head   = nn.Sequential(nn.Linear(num_hiddens, base_dim*8*4*4), group_norm16(base_dim*8*4*4))
        self.b8x8   = DecBlock(base_dim*8, base_dim*4)
        self.b16x16 = DecBlock(base_dim*4, base_dim*2)
        self.b32x32 = DecBlock(base_dim*2, base_dim)


    def forward(self, z):

        x = self.head(z).view(-1, 8*self.base_dim, 4, 4)
        x, i8x8   = self.b8x8(x)
        x, i16x16 = self.b16x16(x)
        x, i32x32 = self.b32x32(x)
        img = i8x8 + i16x16 + i32x32

        loc, scale = torch.chunk(img, 2, dim=1)

        return DiscretizedLogistic(loc, F.softplus(scale) + 1e-6)


class ConvVAE(VAEBase):

    def __init__(self, num_hiddens, base_dim):
        super().__init__()

        self.encoder = ConvEncoder(num_hiddens, base_dim)
        self.decoder = ConvDecoder(num_hiddens, base_dim)

