import torch
import torch.nn.functional as F
import math

def log1mexp(a):
    return torch.where(
            a <= math.log(2.0), 
            torch.log(-torch.expm1(-a)), 
            torch.log1p(-torch.exp(-a)))


class DiscretizedLogistic(object):
    '''
    離散ロジスティック分布
    '''

    def __init__(self, loc, scale):
        self.loc   = loc
        self.scale = scale
        half =  255./2.

        self.min_value = torch.tensor(( -0.5 - half) / half)
        self.max_value = torch.tensor((255.5 - half) / half)
        self.v05       = torch.tensor(0.5 / half)
        self.v1        = torch.tensor(1.0 / half)

        # 0と1が発生すると値が無限大になる
        self.min_cdf         = torch.clamp(self.cdf(self.min_value), torch.finfo(loc.dtype).tiny, 0.99999994)
        self.max_cdf         = torch.clamp(self.cdf(self.max_value), torch.finfo(loc.dtype).tiny, 1.0)

    def log_prob(self, x):
        return -F.softplus(self._z(x-self.v05)) - F.softplus(-self._z(x+self.v05)) + log1mexp(self.v1/self.scale)

    def _z(self, x):
        return (x - self.loc) / self.scale

    def cdf(self, x):
        return torch.sigmoid(self._z(x))

    def rsample(self):
        v = torch.empty_like(self.loc).uniform_(self.min_cdf, self.max_cdf)
        v = torch.log(v) - torch.log1p(-v)
        v = v * self.scale + self.loc
        return torch.clamp(v, self.min_value, self.max_value)
